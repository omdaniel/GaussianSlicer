// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ParityLabMetalRunner",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "parity-lab-metal", targets: ["ParityLabMetalRunner"])
    ],
    targets: [
        .executableTarget(
            name: "ParityLabMetalRunner",
            dependencies: [],
            path: "Sources/ParityLabMetalRunner",
            resources: [
                .process("Shaders.metal")
            ],
            linkerSettings: [
                .linkedFramework("Metal")
            ]
        )
    ]
)
