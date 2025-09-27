// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GaussianSlicer",
    // Ensure the platform is macOS, required for Metal and SwiftUI desktop apps
    platforms: [
        .macOS(.v13) // Requires macOS Ventura (13.0) or later, suitable for M1 Pro
    ],
    products: [
        .executable(
            name: "GaussianSlicer",
            targets: ["GaussianSlicer"]),
    ],
    targets: [
        .executableTarget(
            name: "GaussianSlicer",
            // Explicitly process the Metal shader so it is compiled into the package's default library
            path: "Sources/GaussianSlicer",
            resources: [
                .process("Shaders.metal")
            ]
        ),
    ]
)
