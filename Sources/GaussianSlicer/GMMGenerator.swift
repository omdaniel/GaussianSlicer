// GMMGenerator.swift
import Foundation
import Metal
import simd

// Define data structures matching the Metal shader inputs.
// The layout must be compatible with the MSL definitions.

struct Gaussian3D {
    var mean: SIMD3<Float>
    var covariance: simd_float3x3
    var weight: Float
}

/// Simple SplitMix64-based generator so we can create reproducible seeds per run.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        state = seed &+ 0x9E3779B97F4A7C15
    }
    
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

// Structures for pre-calculated data (Kernel 1 Output)
struct PrecalculatedParams {
    var invCov2D: simd_float2x2
    var normConst2D: Float
    var sigma_n_n: Float
    var meanAdjFactor: SIMD2<Float>
}

// Structures for dynamic data (Kernel 2 Output)
struct DynamicParams {
    var mean2D: SIMD2<Float>
    var combinedFactor: Float
}

class GMMGenerator {
    static func generate(count: Int, meanStdDev: Float, covarianceScale: Float, seed: UInt64?) -> ([Gaussian3D], UInt64) {
        let actualSeed = seed ?? UInt64.random(in: 0..<UInt64.max)
        var rng = SeededGenerator(seed: actualSeed)
        var gaussians = [Gaussian3D]()
        gaussians.reserveCapacity(count)
        var totalWeight: Float = 0.0
        
        for _ in 0..<count {
            let mean = SIMD3<Float>(
                randomNormal(mean: 0.0, stdDev: meanStdDev, using: &rng),
                randomNormal(mean: 0.0, stdDev: meanStdDev, using: &rng),
                randomNormal(mean: 0.0, stdDev: meanStdDev, using: &rng)
            )
            
            let covariance = generateSPDMatrix(using: &rng, scale: covarianceScale)
            let weight = Float.random(in: 0.0...1.0, using: &rng)
            totalWeight += weight
            
            gaussians.append(Gaussian3D(mean: mean, covariance: covariance, weight: weight))
        }
        
        // Normalize weights
        let invTotal = totalWeight > 0 ? (1.0 / totalWeight) : 0.0
        for i in 0..<gaussians.count {
            gaussians[i].weight *= invTotal
        }
        
        return (gaussians, actualSeed)
    }
    
    // Helper to generate Symmetric Positive Definite matrices
    private static func generateSPDMatrix(using rng: inout SeededGenerator, scale: Float) -> simd_float3x3 {
        // Create a random 3x3 matrix A
        let A = simd_float3x3(
            SIMD3<Float>(
                Float.random(in: -1...1, using: &rng),
                Float.random(in: -1...1, using: &rng),
                Float.random(in: -1...1, using: &rng)
            ),
            SIMD3<Float>(
                Float.random(in: -1...1, using: &rng),
                Float.random(in: -1...1, using: &rng),
                Float.random(in: -1...1, using: &rng)
            ),
            SIMD3<Float>(
                Float.random(in: -1...1, using: &rng),
                Float.random(in: -1...1, using: &rng),
                Float.random(in: -1...1, using: &rng)
            )
        )
        
        // C = A^T * A + I*0.5 (Ensure Positive Definite)
        let C = A.transpose * A
        let identity = matrix_identity_float3x3
        let adjustedScale = max(scale, 1e-4)
        let result = (C + (identity * 0.5)) * adjustedScale
        
        return result
    }
    
    private static func randomNormal(mean: Float, stdDev: Float, using rng: inout SeededGenerator) -> Float {
        let epsilon = Float.leastNonzeroMagnitude
        let u1 = max(Float.random(in: 0..<1, using: &rng), epsilon)
        let u2 = Float.random(in: 0..<1, using: &rng)
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2
        let z = radius * cos(theta)
        return mean + stdDev * z
    }
}
