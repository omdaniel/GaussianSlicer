// ContentView.swift
import SwiftUI
import MetalKit

struct ContentView: View {
    // State variable for the slider
    @State private var planeOffset: Float
    // Manage the lifecycle of the renderer using @StateObject
    @StateObject private var renderer: MetalRenderer
    
    init() {
        let config = RuntimeConfig.shared
        let initialOffset = max(config.gridMin, min(0.0, config.gridMax))
        _planeOffset = State(initialValue: initialOffset)
        
        // The App entry point already verified Metal support, but we ensure the device exists here.
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device initialization failed unexpectedly.")
        }
        // Initialize the renderer StateObject
        _renderer = StateObject(wrappedValue: MetalRenderer(device: device))
    }

    var body: some View {
        VStack(spacing: 5) {
             // Header with Performance Metrics
            HStack {
                Text("3D GMM Slicing (Metal Accelerated)")
                    .font(.headline)
                Spacer()
                Text("N=\(renderer.NUM_DISTRIBUTIONS) | Grid=\(renderer.GRID_RESOLUTION)x\(renderer.GRID_RESOLUTION)")
                    .font(.subheadline)
                Text(String(format: "Frame Time: %.2f ms", renderer.frameTime))
                    .font(.subheadline)
                    .monospacedDigit()
                    .padding(5)
                    .background(Color.secondary.opacity(0.2))
                    .cornerRadius(5)
            }
            .padding(.horizontal)
            .padding(.top, 10)

            
            // Host the Metal View
            MetalView(renderer: renderer)
                .frame(minWidth: 512, maxWidth: .infinity, minHeight: 512, maxHeight: .infinity)
                .aspectRatio(1.0, contentMode: .fit)
                .border(Color.gray, width: 1)
                .onAppear {
                    renderer.currentOffset = planeOffset
                }

            // Interactive Slider
            HStack {
                Text(String(format: "Offset: %.2f", planeOffset))
                    .frame(width: 100, alignment: .leading)
                // Slider updates the planeOffset state variable
                Slider(value: $planeOffset, in: renderer.GRID_MIN...renderer.GRID_MAX)
                    .padding()
                    .onChange(of: planeOffset) { newValue in
                        // Update the renderer's published property when the slider moves.
                        // This triggers the MTKView's draw loop via the ObservableObject mechanism.
                        renderer.currentOffset = newValue
                    }
            }
            .padding(.horizontal)
        }
        .padding()
    }
}

// Helper view to bridge MTKView into SwiftUI for macOS
struct MetalView: NSViewRepresentable {
    // Observe the renderer object for changes
    @ObservedObject var renderer: MetalRenderer
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.delegate = renderer
        mtkView.device = renderer.device
        // Configure the view for continuous rendering (driven by the draw loop)
        mtkView.isPaused = false
        mtkView.enableSetNeedsDisplay = false
        // Set the pixel format (must match the render pipeline definition in MetalRenderer.swift)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Updates are driven by the renderer's draw loop and the ObservableObject mechanism
    }
}
