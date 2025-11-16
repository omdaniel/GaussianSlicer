// MetalRenderer.swift
import Combine
import Foundation
import MetalKit
import simd
import Darwin
import AppKit

// Configuration structure matching the Metal shader definition
struct Config {
    var numDistributions: UInt32
    var rotationMatrix: simd_float3x3
    var planeNormal: SIMD3<Float>
    var planeOffset: Float
    var gridMin: Float
    var gridMax: Float
}

struct VisualizationConfig {
    var colormapIndex: UInt32
    var invert: UInt32
    var logScale: UInt32
    var colorLevels: UInt32
    var densityMin: Float
    var densityMax: Float
    var outlineWidth: Float
    var filterMode: UInt32
}

class MetalRenderer: NSObject, MTKViewDelegate, ObservableObject {
    
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    
    private var settings: RendererSettings
    @Published private(set) var numDistributions: Int
    @Published private(set) var gridResolution: Int
    @Published private(set) var gridMin: Float
    @Published private(set) var gridMax: Float

    // Buffers
    var gaussianBuffer: MTLBuffer!
    var precalcBuffer: MTLBuffer!
    var dynamicBuffer: MTLBuffer!
    var config: Config!
    var visualizationConfig: VisualizationConfig
    
    // Pipelines
    var precalcPipeline: MTLComputePipelineState!
    var updatePipeline: MTLComputePipelineState!
    var evaluationPipeline: MTLComputePipelineState!
    var renderPipeline: MTLRenderPipelineState!
    
    // Output
    var densityTexture: MTLTexture!
    private var lastCommandBuffer: MTLCommandBuffer?
    private var captureFrameURL: URL?
    private var captureTexture: MTLTexture?

    // State
    @Published var currentOffset: Float = 0.0
    @Published var frameTime: Double = 0.0 // Performance metric
    var isInitialized = false
    private let initializationTimeoutSeconds: TimeInterval = 10.0
    @Published var isExporting = false
    private let gaussianPlyURL: URL?

    init(device: MTLDevice, settings: RendererSettings) {
        self.device = device
        self.commandQueue = self.device.makeCommandQueue()!
        self.settings = settings
        self.numDistributions = settings.numDistributions
        self.gridResolution = settings.gridResolution
        self.gridMin = settings.gridMin
        self.gridMax = settings.gridMax

        visualizationConfig = VisualizationConfig(
            colormapIndex: settings.colormap.rawValue,
            invert: settings.invertColormap ? 1 : 0,
            logScale: settings.useLogScale ? 1 : 0,
            colorLevels: settings.colorLevels,
            densityMin: settings.densityMin,
            densityMax: settings.densityMax,
            outlineWidth: settings.outlineWidth,
            filterMode: settings.filterMode
        )
        gaussianPlyURL = RuntimeConfig.shared.gaussianPlyURL

        captureFrameURL = RuntimeConfig.shared.captureFrameURL

        super.init()

        // Abort if initialization logits never finish (e.g., shader compilation failure).
        DispatchQueue.main.asyncAfter(deadline: .now() + initializationTimeoutSeconds) { [weak self] in
            guard let self = self else { return }
            if !self.isInitialized {
                fputs("Initialization timed out after \(self.initializationTimeoutSeconds)s. Exiting.\n", stderr)
                exit(EXIT_FAILURE)
            }
        }

        setupMetal()
        generateData()
        // Pipelines must be setup after data, as SwiftPM compiles Metal on first access.
        if setupPipelines() {
            setupTexture()
            runInitialPrecalculation()
        } else {
            print("Pipeline setup failed. Initialization halted.")
        }
    }
    
    func setupMetal() {
        let planeNormal = settings.planeNormal
        let rotationMatrix = MetalRenderer.getRotationMatrix(normal: planeNormal)
        let clampedOffset = max(min(currentOffset, gridMax), gridMin)
        currentOffset = clampedOffset

        self.config = Config(
            numDistributions: UInt32(max(0, min(numDistributions, Int(UInt32.max)))),
            rotationMatrix: rotationMatrix,
            planeNormal: planeNormal,
            planeOffset: clampedOffset,
            gridMin: gridMin,
            gridMax: gridMax
        )
    }
    
    func generateData() {
        let startTime = CACurrentMediaTime()
        let gaussianData: [Gaussian3D]
        var generationNote = ""

        if let plyURL = gaussianPlyURL {
            do {
                gaussianData = try GaussianSplatPLY.load(url: plyURL)
                print("Loaded \(gaussianData.count) Gaussian splats from \(plyURL.path)")
            } catch {
                print("Failed to load Gaussian PLY '\(plyURL.path)': \(error). Falling back to procedural generation.")
                let (generated, seed) = GMMGenerator.generate(
                    count: settings.numDistributions,
                    meanStdDev: settings.meanStdDev,
                    covarianceScale: settings.covarianceScale,
                    seed: settings.seed
                )
                gaussianData = generated
                generationNote = " (seed: \(seed))"
            }
        } else {
            let (generated, seed) = GMMGenerator.generate(
                count: numDistributions,
                meanStdDev: settings.meanStdDev,
                covarianceScale: settings.covarianceScale,
                seed: settings.seed
            )
            gaussianData = generated
            generationNote = " (seed: \(seed))"
        }

        numDistributions = gaussianData.count
        settings.numDistributions = gaussianData.count
        config.numDistributions = UInt32(max(0, min(numDistributions, Int(UInt32.max))))

        let endTime = CACurrentMediaTime()
        let elapsed = endTime - startTime
        print("Data preparation time: \(String(format: "%.4f", elapsed))s\(generationNote)")

        let gaussianSize = MemoryLayout<Gaussian3D>.stride * gaussianData.count
        let precalcSize = MemoryLayout<PrecalculatedParams>.stride * gaussianData.count
        let dynamicSize = MemoryLayout<DynamicParams>.stride * gaussianData.count

        gaussianBuffer = device.makeBuffer(bytes: gaussianData, length: gaussianSize, options: .storageModeShared)
        precalcBuffer = device.makeBuffer(length: precalcSize, options: .storageModePrivate)
        dynamicBuffer = device.makeBuffer(length: dynamicSize, options: .storageModePrivate)
        print("Buffers initialized.")
    }

    func setupPipelines() -> Bool {
        do {
            let library: MTLLibrary
            if let defaultLibrary = device.makeDefaultLibrary() {
                library = defaultLibrary
            } else if let bundledLibrary = try? device.makeDefaultLibrary(bundle: .module) {
                library = bundledLibrary
            } else {
                guard let shaderURL = Bundle.module.url(forResource: "Shaders", withExtension: "metal") else {
                    print("Error: Could not locate Shaders.metal resource. Ensure it is included in Package.swift.")
                    return false
                }
                let shaderSource = try String(contentsOf: shaderURL)
                library = try device.makeLibrary(source: shaderSource, options: nil)
            }
            
            // Compute Pipelines
            let precalcFunc = library.makeFunction(name: "precalculateKernel")!
            precalcPipeline = try device.makeComputePipelineState(function: precalcFunc)
            
            let updateFunc = library.makeFunction(name: "updateParamsKernel")!
            updatePipeline = try device.makeComputePipelineState(function: updateFunc)
            
            let evalFunc = library.makeFunction(name: "evaluationKernel")!
            evaluationPipeline = try device.makeComputePipelineState(function: evalFunc)
            
            // Render Pipeline
            let renderDescriptor = MTLRenderPipelineDescriptor()
            renderDescriptor.vertexFunction = library.makeFunction(name: "vertexShader")
            renderDescriptor.fragmentFunction = library.makeFunction(name: "fragmentShader")
            // Must match the MTKView pixel format (set in ContentView.swift)
            renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            
            renderPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)
            return true
            
        } catch {
            print("Error: Unable to create pipelines: \(error)")
            return false
        }
    }
    
    func setupTexture() {
        let resolution = max(1, gridResolution)
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, // 32-bit float for density precision
            width: resolution,
            height: resolution,
            mipmapped: false
        )
        // Written by compute shader, read by fragment shader
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        // Optimized storage mode for Apple Silicon (GPU only)
        textureDescriptor.storageMode = .private
        densityTexture = device.makeTexture(descriptor: textureDescriptor)
    }

    func apply(settings newSettings: RendererSettings) {
        lastCommandBuffer?.waitUntilCompleted()
        lastCommandBuffer = nil
        isInitialized = false

        settings = newSettings
        if gaussianPlyURL == nil {
            numDistributions = newSettings.numDistributions
        } else {
            settings.numDistributions = numDistributions
        }
        gridResolution = newSettings.gridResolution
        gridMin = newSettings.gridMin
        gridMax = newSettings.gridMax
        visualizationConfig = VisualizationConfig(
            colormapIndex: newSettings.colormap.rawValue,
            invert: newSettings.invertColormap ? 1 : 0,
            logScale: newSettings.useLogScale ? 1 : 0,
            colorLevels: newSettings.colorLevels,
            densityMin: newSettings.densityMin,
            densityMax: newSettings.densityMax,
            outlineWidth: newSettings.outlineWidth,
            filterMode: newSettings.filterMode
        )

        currentOffset = max(min(currentOffset, gridMax), gridMin)
        setupMetal()
        generateData()
        setupTexture()
        runInitialPrecalculation()
    }
    
    // MARK: - Initialization Kernel Execution
    
    func runInitialPrecalculation() {
        print("Running Pre-calculation Kernel (K1)...")
        let startTime = CACurrentMediaTime()

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        runComputeKernel(encoder: computeCommandEncoder, pipeline: precalcPipeline, count: numDistributions)
        
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        // Crucial: Wait for K1 to complete before proceeding, as K2 depends on it.
        commandBuffer.waitUntilCompleted()
        
        let endTime = CACurrentMediaTime()
        print("K1 Complete. Time: \(String(format: "%.4f", endTime - startTime))s")
        isInitialized = true
    }

    // MARK: - MTKViewDelegate Methods
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    // The main rendering loop (Called continuously by MTKView)
    func draw(in view: MTKView) {
        // Do not start the main loop until initialization is finished
        if !isInitialized || isExporting { return }
        
        let startTime = CACurrentMediaTime()

        // Update config based on the published offset from SwiftUI (ObservableObject)
        config.planeOffset = currentOffset
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        // MARK: Compute Pass
        if let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() {
            // Kernel 2: Update Parameters (O(N))
            runComputeKernel(encoder: computeCommandEncoder, pipeline: updatePipeline, count: numDistributions)
            
            // Kernel 3: Evaluation (O(N*G))
            runEvaluationKernel(encoder: computeCommandEncoder)
            
            computeCommandEncoder.endEncoding()
        }
    
        // MARK: Render Pass (Visualization)
        if let renderPassDescriptor = view.currentRenderPassDescriptor,
           let drawable = view.currentDrawable {
            encodeVisualization(renderPassDescriptor, commandBuffer: commandBuffer)
            commandBuffer.present(drawable)
        }

        if let captureURL = captureFrameURL {
            encodeCapturePass(commandBuffer: commandBuffer, url: captureURL)
        }
        
        // Measure execution time and update the UI asynchronously
        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self else { return }
            self.lastCommandBuffer = nil
            let endTime = CACurrentMediaTime()
            let frameTimeMs = (endTime - startTime) * 1000.0
            DispatchQueue.main.async {
                self.frameTime = frameTimeMs
            }
        }

        lastCommandBuffer = commandBuffer
        commandBuffer.commit()
    }
    
    // MARK: - Kernel Execution Helpers

    // Sets common buffers for all compute kernels
    func setCommonBuffers(encoder: MTLComputeCommandEncoder) {
        // Pass the config struct by value (efficient for small structs)
        encoder.setBytes(&config, length: MemoryLayout<Config>.stride, index: 0)
        encoder.setBuffer(gaussianBuffer, offset: 0, index: 1)
        encoder.setBuffer(precalcBuffer, offset: 0, index: 2)
        encoder.setBuffer(dynamicBuffer, offset: 0, index: 3)
    }

    private func encodeVisualization(_ descriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        renderEncoder.setRenderPipelineState(renderPipeline)
        renderEncoder.setFragmentTexture(densityTexture, index: 0)
        var vizConfig = visualizationConfig
        renderEncoder.setFragmentBytes(&vizConfig, length: MemoryLayout<VisualizationConfig>.stride, index: 0)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        renderEncoder.endEncoding()
    }

    private func encodeCapturePass(commandBuffer: MTLCommandBuffer, url: URL) {
        let resolution = gridResolution
        guard let texture = ensureCaptureTexture(resolution: resolution) else { return }
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = texture
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
        encodeVisualization(descriptor, commandBuffer: commandBuffer)
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.writeCaptureTexture(texture: texture, resolution: resolution, url: url)
        }
    }

    private func ensureCaptureTexture(resolution: Int) -> MTLTexture? {
        if let texture = captureTexture,
           texture.width == resolution,
           texture.height == resolution {
            return texture
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: resolution, height: resolution, mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        captureTexture = device.makeTexture(descriptor: descriptor)
        return captureTexture
    }

    private func writeCaptureTexture(texture: MTLTexture, resolution: Int, url: URL) {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * resolution
        var data = Data(count: bytesPerRow * resolution)
        data.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            let region = MTLRegionMake2D(0, 0, resolution, resolution)
            texture.getBytes(baseAddress, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let alphaInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(CGBitmapInfo(rawValue: alphaInfo))
        guard let provider = CGDataProvider(data: data as CFData),
              let cgImage = CGImage(
                  width: resolution,
                  height: resolution,
                  bitsPerComponent: 8,
                  bitsPerPixel: 32,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: bitmapInfo,
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              ) else {
            print("Failed to create capture CGImage")
            return
        }

        let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
        guard let pngData = bitmapRep.representation(using: NSBitmapImageRep.FileType.png, properties: [:]) else {
            print("Failed to encode capture PNG")
            return
        }

        do {
            try pngData.write(to: url)
            captureFrameURL = nil
            DispatchQueue.main.async {
                exit(EXIT_SUCCESS)
            }
        } catch {
            print("Failed to save capture: \(error)")
        }
    }

    // Helper for 1D compute kernels (K1, K2)
    func runComputeKernel(encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int) {
        encoder.setComputePipelineState(pipeline)
        setCommonBuffers(encoder: encoder)
        
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        // Determine optimal thread group size for 1D dispatch
        let threadGroupWidth = pipeline.threadExecutionWidth
        let threadGroupSize = MTLSize(width: threadGroupWidth, height: 1, depth: 1)
        
        // Use dispatchThreads for modern Metal (optimized for Apple Silicon)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    }
    
    // Run compute for a specific plane offset and copy the density into a shared texture for readback
    func computeSlice(at offset: Float, into readbackTexture: MTLTexture) -> Bool {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return false }

        config.planeOffset = offset
        if let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() {
            runComputeKernel(encoder: computeCommandEncoder, pipeline: updatePipeline, count: numDistributions)
            runEvaluationKernel(encoder: computeCommandEncoder)
            computeCommandEncoder.endEncoding()
        }

        if let blit = commandBuffer.makeBlitCommandEncoder() {
            let srcSize = MTLSize(width: densityTexture.width, height: densityTexture.height, depth: 1)
            blit.copy(from: densityTexture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: srcSize, to: readbackTexture, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return true
    }

    // Helper for 2D compute kernel (K3)
    func runEvaluationKernel(encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(evaluationPipeline)
        setCommonBuffers(encoder: encoder)
        encoder.setTexture(densityTexture, index: 0)
        
        let resolution = max(1, gridResolution)
        let gridSize = MTLSize(width: resolution, height: resolution, depth: 1)
        
        // Determine optimal 2D thread group size
        let w = evaluationPipeline.threadExecutionWidth
        // Ensure the total threads do not exceed the maximum allowed
        let h = max(1, min(evaluationPipeline.maxTotalThreadsPerThreadgroup / w, resolution))
        let threadGroupSize = MTLSize(width: w, height: h, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    }
    
    // MARK: - Utility
    
    // Helper to calculate the rotation matrix R = [u, v, n] (rows)
    static func getRotationMatrix(normal n: SIMD3<Float>) -> simd_float3x3 {
        let n_norm = simd_normalize(n)
        
        // Find an arbitrary vector not collinear with n (robust method)
        var arbitraryVec = SIMD3<Float>(0, 0, 0)
        // Find the component with the smallest absolute value
        let absN = abs(n_norm)
        var minIndex = 0
        if absN.y < absN.x { minIndex = 1 }
        if absN.z < absN[minIndex] { minIndex = 2 }
        
        arbitraryVec[minIndex] = 1.0
        
        var u = simd_cross(n_norm, arbitraryVec)
        
        // Handle potential collinearity robustly
        if simd_length(u) < 1e-6 {
            arbitraryVec = SIMD3<Float>(0, 0, 0)
            arbitraryVec[(minIndex + 1) % 3] = 1.0
            u = simd_cross(n_norm, arbitraryVec)
        }
        
        u = simd_normalize(u)
        let v = simd_normalize(simd_cross(n_norm, u))
        
        // Construct rotation matrix (rows are u, v, n)
        return simd_float3x3(rows: [u, v, n_norm])
    }

    // Expose current local axes based on the active plane normal
    var axesUVN: (u: SIMD3<Float>, v: SIMD3<Float>, n: SIMD3<Float>) {
        let R = MetalRenderer.getRotationMatrix(normal: settings.planeNormal)
        // R was constructed with rows [u, v, n]
        // Simd matrices are column-major; there isn't direct row access, so reconstruct
        // u = row 0 = (R[0][0], R[1][0], R[2][0])
        let u = SIMD3<Float>(R[0][0], R[1][0], R[2][0])
        let v = SIMD3<Float>(R[0][1], R[1][1], R[2][1])
        let n = SIMD3<Float>(R[0][2], R[1][2], R[2][2])
        return (u, v, n)
    }
}
