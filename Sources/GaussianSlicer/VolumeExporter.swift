import Foundation
import Metal
import AppKit

final class VolumeExporter {
    static func exportVolume(renderer: MetalRenderer, parentWindow: NSWindow?, normalizedLog01: Bool) {
        let panel = NSSavePanel()
        panel.title = "Export Volume"
        panel.allowedFileTypes = ["mhd", "raw"]
        panel.nameFieldStringValue = "density.mhd"
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false

        panel.beginSheetModal(for: parentWindow ?? NSApp.mainWindow ?? NSWindow()) { response in
            guard response == .OK, let url = panel.url else { return }
            Task.detached { await self.performExport(renderer: renderer, destination: url, normalizedLog01: normalizedLog01) }
        }
    }

    private static func performExport(renderer: MetalRenderer, destination url: URL, normalizedLog01: Bool) async {
        let N = renderer.gridResolution
        let width = N
        let height = N
        let depth = N
        let totalCount = width * height * depth
        var volume = [Float](repeating: 0, count: totalCount)

        let device = renderer.device
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.storageMode = .shared
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        guard let readbackTexture = device.makeTexture(descriptor: textureDescriptor) else { return }

        let range = renderer.gridMax - renderer.gridMin
        let denom = max(1, N - 1)

        await MainActor.run { renderer.isExporting = true }

        for zi in 0..<depth {
            let t = Float(zi) / Float(denom)
            let offset = renderer.gridMin + t * range
            if !renderer.computeSlice(at: offset, into: readbackTexture) {
                continue
            }

            // Read back row by row
            let rowBytes = width * MemoryLayout<Float>.size
            var sliceBuffer = [Float](repeating: 0, count: width * height)
            sliceBuffer.withUnsafeMutableBytes { ptr in
                for y in 0..<height {
                    let region = MTLRegionMake2D(0, y, width, 1)
                    readbackTexture.getBytes(ptr.baseAddress!.advanced(by: y * rowBytes), bytesPerRow: rowBytes, from: region, mipmapLevel: 0)
                }
            }

            if normalizedLog01 {
                // Normalize using log scale into [0,1] based on current visualization config
                let viz = renderer.visualizationConfig
                let minPos: Float = 1e-12
                let vMin = max(viz.densityMin, minPos)
                let vMax = max(viz.densityMax, vMin + minPos)
                let logVMin = log(vMin)
                let logDen = log(vMax) - logVMin
                let invLogDen: Float = (abs(logDen) > minPos) ? 1.0 / logDen : 0.0
                for i in 0..<sliceBuffer.count {
                    let d = max(sliceBuffer[i], minPos)
                    var t = (log(d) - logVMin) * invLogDen
                    if viz.invert != 0 { t = 1.0 - t }
                    sliceBuffer[i] = max(0.0, min(1.0, t))
                }
            }

            // Copy into volume at z = zi (z-major slices, row-major per slice)
            let base = zi * width * height
            _ = sliceBuffer.withUnsafeBytes { src in
                volume.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!.advanced(by: base * MemoryLayout<Float>.size), src.baseAddress!, sliceBuffer.count * MemoryLayout<Float>.size)
                }
            }
        }

        await MainActor.run { renderer.isExporting = false }

        // Write RAW + MHD header next to it
        let rawURL: URL
        let mhdURL: URL
        if url.pathExtension.lowercased() == "mhd" {
            mhdURL = url
            rawURL = url.deletingPathExtension().appendingPathExtension("raw")
        } else if url.pathExtension.lowercased() == "raw" {
            rawURL = url
            mhdURL = url.deletingPathExtension().appendingPathExtension("mhd")
        } else {
            mhdURL = url.appendingPathExtension("mhd")
            rawURL = url.appendingPathExtension("raw")
        }

        // RAW (little-endian float32)
        do {
            try Data(bytes: &volume, count: volume.count * MemoryLayout<Float>.size).write(to: rawURL)
        } catch {
            print("Error writing RAW volume: \(error)")
            return
        }

        // Header (MHD)
        let spacing = Double(range) / Double(denom)
        let header = """
        ObjectType = Image
        NDims = 3
        DimSize = \(width) \(height) \(depth)
        ElementType = MET_FLOAT
        ElementSpacing = \(spacing) \(spacing) \(spacing)
        ElementByteOrderMSB = False
        ElementDataFile = \(rawURL.lastPathComponent)
        """.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        do {
            try header.data(using: .utf8)?.write(to: mhdURL)
        } catch {
            print("Error writing MHD header: \(error)")
        }
        print("Export complete: \(mhdURL.path) + \(rawURL.path)")
    }

    // MARK: - OpenVDB Export (external tool)
    static func exportOpenVDB(renderer: MetalRenderer, parentWindow: NSWindow?, normalizedLog01: Bool) {
        let panel = NSSavePanel()
        panel.title = "Export OpenVDB"
        panel.allowedFileTypes = ["vdb"]
        panel.nameFieldStringValue = "density.vdb"
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false

        panel.beginSheetModal(for: parentWindow ?? NSApp.mainWindow ?? NSWindow()) { response in
            guard response == .OK, let url = panel.url else { return }
            Task.detached { await self.performExportOpenVDB(renderer: renderer, destination: url, normalizedLog01: normalizedLog01) }
        }
    }

    private static func locateVDBWriter() -> String? {
        let fm = FileManager.default
        // Search candidates relative to current working directory and app bundle
        var candidates: [String] = []
        let cwd = fm.currentDirectoryPath
        candidates.append((cwd as NSString).appendingPathComponent("Tools/vdb_writer"))
        candidates.append((cwd as NSString).appendingPathComponent("vdb_writer"))
        // Also try alongside the executable
        if let exeURL = Bundle.main.executableURL {
            candidates.append(exeURL.deletingLastPathComponent().appendingPathComponent("vdb_writer").path)
        }
        for p in candidates {
            if fm.isExecutableFile(atPath: p) { return p }
        }
        return nil
    }

    private static func performExportOpenVDB(renderer: MetalRenderer, destination url: URL, normalizedLog01: Bool) async {
        // First compute the RAW volume into a temporary file
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let rawURL = tempDir.appendingPathComponent("density_\(UUID().uuidString).raw")
        let N = renderer.gridResolution
        let width = N
        let height = N
        let depth = N
        let totalCount = width * height * depth
        var volume = [Float](repeating: 0, count: totalCount)

        let device = renderer.device
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.storageMode = .shared
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        guard let readbackTexture = device.makeTexture(descriptor: textureDescriptor) else { return }

        let range = renderer.gridMax - renderer.gridMin
        let denom = max(1, N - 1)

        await MainActor.run { renderer.isExporting = true }

        for zi in 0..<depth {
            let t = Float(zi) / Float(denom)
            let offset = renderer.gridMin + t * range
            if !renderer.computeSlice(at: offset, into: readbackTexture) {
                continue
            }

            let rowBytes = width * MemoryLayout<Float>.size
            var sliceBuffer = [Float](repeating: 0, count: width * height)
            sliceBuffer.withUnsafeMutableBytes { ptr in
                for y in 0..<height {
                    let region = MTLRegionMake2D(0, y, width, 1)
                    readbackTexture.getBytes(ptr.baseAddress!.advanced(by: y * rowBytes), bytesPerRow: rowBytes, from: region, mipmapLevel: 0)
                }
            }

            if normalizedLog01 {
                let viz = renderer.visualizationConfig
                let minPos: Float = 1e-12
                let vMin = max(viz.densityMin, minPos)
                let vMax = max(viz.densityMax, vMin + minPos)
                let logVMin = log(vMin)
                let logDen = log(vMax) - logVMin
                let invLogDen: Float = (abs(logDen) > minPos) ? 1.0 / logDen : 0.0
                for i in 0..<sliceBuffer.count {
                    let d = max(sliceBuffer[i], minPos)
                    var t = (log(d) - logVMin) * invLogDen
                    if viz.invert != 0 { t = 1.0 - t }
                    sliceBuffer[i] = max(0.0, min(1.0, t))
                }
            }
            let base = zi * width * height
            _ = sliceBuffer.withUnsafeBytes { src in
                volume.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!.advanced(by: base * MemoryLayout<Float>.size), src.baseAddress!, sliceBuffer.count * MemoryLayout<Float>.size)
                }
            }
        }

        await MainActor.run { renderer.isExporting = false }

        do {
            try Data(bytes: &volume, count: volume.count * MemoryLayout<Float>.size).write(to: rawURL)
        } catch {
            print("Error writing temp RAW volume: \(error)")
            return
        }

        guard let tool = locateVDBWriter() else {
            await MainActor.run {
                let alert = NSAlert()
                alert.messageText = "OpenVDB writer not found"
                alert.informativeText = "Please build Tools/vdb_writer using scripts/build_vdb_writer.sh (requires Homebrew openvdb)."
                alert.alertStyle = .warning
                alert.runModal()
            }
            return
        }

        let spacing = Double(range) / Double(denom)
        let task = Process()
        task.executableURL = URL(fileURLWithPath: tool)
        task.arguments = [
            "--raw", rawURL.path,
            "--dim", String(width), String(height), String(depth),
            "--spacing", String(spacing), String(spacing), String(spacing),
            "--out", url.path
        ]

        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            print("Failed running vdb_writer: \(error)")
        }

        // Cleanup temporary RAW
        try? FileManager.default.removeItem(at: rawURL)
    }
}
