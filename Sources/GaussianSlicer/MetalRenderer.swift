// MetalRenderer.swift
import Foundation
import MetalKit
import simd
import Darwin

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
}

class MetalRenderer: NSObject, MTKViewDelegate, ObservableObject {
    
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    
    let runtimeConfig = RuntimeConfig.shared
    let NUM_DISTRIBUTIONS: Int
    let GRID_RESOLUTION: Int
    let GRID_MIN: Float
    let GRID_MAX: Float

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

    // State
    @Published var currentOffset: Float = 0.0
    @Published var frameTime: Double = 0.0 // Performance metric
    var isInitialized = false
    private let initializationTimeoutSeconds: TimeInterval = 10.0

    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = self.device.makeCommandQueue()!

        NUM_DISTRIBUTIONS = runtimeConfig.numDistributions
        GRID_RESOLUTION = runtimeConfig.gridResolution
        GRID_MIN = runtimeConfig.gridMin
        GRID_MAX = runtimeConfig.gridMax
        visualizationConfig = VisualizationConfig(
            colormapIndex: runtimeConfig.colormap.rawValue,
            invert: runtimeConfig.invertColormap ? 1 : 0,
            logScale: runtimeConfig.useLogScale ? 1 : 0,
            colorLevels: runtimeConfig.colorLevels,
            densityMin: runtimeConfig.densityMin,
            densityMax: runtimeConfig.densityMax,
            outlineWidth: runtimeConfig.outlineWidth
        )

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
        let planeNormal = runtimeConfig.planeNormal
        // Calculate R such that rows are [u, v, n]
        let rotationMatrix = MetalRenderer.getRotationMatrix(normal: planeNormal)
        
        // Initial configuration setup
        self.config = Config(
            numDistributions: UInt32(max(0, min(NUM_DISTRIBUTIONS, Int(UInt32.max)))),
            rotationMatrix: rotationMatrix,
            planeNormal: planeNormal,
            planeOffset: 0.0,
            gridMin: GRID_MIN,
            gridMax: GRID_MAX
        )
    }
    
    func generateData() {
        print("Generating \(NUM_DISTRIBUTIONS) distributions...")
        let startTime = CACurrentMediaTime()
        let (gaussians, seed) = GMMGenerator.generate(
            count: NUM_DISTRIBUTIONS,
            meanStdDev: runtimeConfig.meanStdDev,
            covarianceScale: runtimeConfig.covarianceScale,
            seed: runtimeConfig.seed
        )
        let endTime = CACurrentMediaTime()
        print("Data generation time: \(String(format: "%.4f", endTime - startTime))s (seed: \(seed))")

        
        // Create Buffers using MemoryLayout<T>.stride
        let gaussianSize = MemoryLayout<Gaussian3D>.stride * NUM_DISTRIBUTIONS
        let precalcSize = MemoryLayout<PrecalculatedParams>.stride * NUM_DISTRIBUTIONS
        let dynamicSize = MemoryLayout<DynamicParams>.stride * NUM_DISTRIBUTIONS
        
        // Input buffer: Shared mode for CPU initialization
        gaussianBuffer = device.makeBuffer(bytes: gaussians, length: gaussianSize, options: .storageModeShared)
        
        // Intermediate buffers: Private mode (GPU only) for optimization on Apple Silicon
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
        let resolution = max(1, GRID_RESOLUTION)
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
    
    // MARK: - Initialization Kernel Execution
    
    func runInitialPrecalculation() {
        print("Running Pre-calculation Kernel (K1)...")
        let startTime = CACurrentMediaTime()

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        runComputeKernel(encoder: computeCommandEncoder, pipeline: precalcPipeline, count: NUM_DISTRIBUTIONS)
        
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
        if !isInitialized { return }
        
        let startTime = CACurrentMediaTime()

        // Update config based on the published offset from SwiftUI (ObservableObject)
        config.planeOffset = currentOffset
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        // MARK: Compute Pass
        if let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() {
            // Kernel 2: Update Parameters (O(N))
            runComputeKernel(encoder: computeCommandEncoder, pipeline: updatePipeline, count: NUM_DISTRIBUTIONS)
            
            // Kernel 3: Evaluation (O(N*G))
            runEvaluationKernel(encoder: computeCommandEncoder)
            
            computeCommandEncoder.endEncoding()
        }
    
        // MARK: Render Pass (Visualization)
        if let renderPassDescriptor = view.currentRenderPassDescriptor,
           let drawable = view.currentDrawable {
            
            if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                renderEncoder.setRenderPipelineState(renderPipeline)
                // Pass the computed density texture to the fragment shader
                renderEncoder.setFragmentTexture(densityTexture, index: 0)
                var vizConfig = visualizationConfig
                renderEncoder.setFragmentBytes(&vizConfig, length: MemoryLayout<VisualizationConfig>.stride, index: 0)
                // Draw a full-screen quad (6 vertices, 2 triangles)
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                renderEncoder.endEncoding()
            }
            // Present the final visualization
            commandBuffer.present(drawable)
        }
        
        // Measure execution time and update the UI asynchronously
        commandBuffer.addCompletedHandler { [weak self] _ in
            DispatchQueue.main.async {
                let endTime = CACurrentMediaTime()
                self?.frameTime = (endTime - startTime) * 1000.0 // Convert to ms
            }
        }

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
    
    // Helper for 2D compute kernel (K3)
    func runEvaluationKernel(encoder: MTLComputeCommandEncoder) {
        encoder.setComputePipelineState(evaluationPipeline)
        setCommonBuffers(encoder: encoder)
        encoder.setTexture(densityTexture, index: 0)
        
        let resolution = max(1, GRID_RESOLUTION)
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
}
