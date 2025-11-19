import SwiftUI
import AppKit
import Foundation

// The entry point for the macOS SwiftUI Application
@main
struct GaussianSlicerApp: App {
    init() {
        let runtime = RuntimeConfig.shared
        if runtime.shouldExit {
            exit(EXIT_SUCCESS)
        }
        if let exportURL = runtime.exportVolumeURL {
            guard let device = MTLCreateSystemDefaultDevice() else {
                fatalError("Metal is required for headless export.")
            }
            let appSettings = AppSettings(config: runtime)
            let rendererSettings = appSettings.makeRendererSettings()
            let renderer = MetalRenderer(device: device, settings: rendererSettings)
            VolumeExporter.exportVolumeHeadless(
                renderer: renderer,
                destination: exportURL,
                normalizedLog01: runtime.exportLogNormalized
            )
            exit(EXIT_SUCCESS)
        }
        // Ensure the app becomes a regular, frontmost app when launched from Terminal
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.async {
            NSApplication.shared.activate(ignoringOtherApps: true)
            NSApplication.shared.windows.forEach { $0.makeKeyAndOrderFront(nil) }
        }
        
        if let exitDelay = runtime.exitAfterMs {
            DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(Int(exitDelay))) {
                print("Exiting after \(exitDelay)ms as requested.")
                exit(EXIT_SUCCESS)
            }
        }
    }
    
    var body: some Scene {
        WindowGroup {
            // Check if the device supports Metal before launching the main view
            if MTLCreateSystemDefaultDevice() != nil {
                ContentView()
            } else {
                Text("Metal is required to run this application.\nThis application is optimized for Apple Silicon.")
                    .frame(width: 400, height: 200)
                    .multilineTextAlignment(.center)
                    .padding()
            }
        }
        // Set a default window size
        .defaultSize(width: 850, height: 950)
    }
}
