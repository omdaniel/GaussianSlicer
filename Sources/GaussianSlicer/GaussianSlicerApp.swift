import SwiftUI
import AppKit

// The entry point for the macOS SwiftUI Application
@main
struct GaussianSlicerApp: App {
    init() {
        if RuntimeConfig.shared.shouldExit {
            exit(EXIT_SUCCESS)
        }
        // Ensure the app becomes a regular, frontmost app when launched from Terminal
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.async {
            NSApplication.shared.activate(ignoringOtherApps: true)
            NSApplication.shared.windows.forEach { $0.makeKeyAndOrderFront(nil) }
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
