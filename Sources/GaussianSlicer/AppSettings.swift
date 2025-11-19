import Combine
import Foundation
import simd

struct RendererSettings {
    var numDistributions: Int
    var gridResolution: Int
    var gridMin: Float
    var gridMax: Float
    var meanStdDev: Float
    var covarianceScale: Float
    var seed: UInt64?
    var planeNormal: SIMD3<Float>
    var colormap: ColormapOption
    var invertColormap: Bool
    var useLogScale: Bool
    var densityMin: Float
    var densityMax: Float
    var colorLevels: UInt32
    var outlineWidth: Float
    var filterMode: UInt32
}

final class AppSettings: ObservableObject {
    @Published var numDistributionsText: String
    @Published var gridResolutionText: String
    @Published var gridMinText: String
    @Published var gridMaxText: String
    @Published var meanStdDevText: String
    @Published var covarianceScaleText: String
    @Published var seedString: String
    @Published var planeNormalXText: String
    @Published var planeNormalYText: String
    @Published var planeNormalZText: String
    @Published var colormap: ColormapOption
    @Published var invertColormap: Bool
    @Published var useLogScale: Bool
    @Published var densityMinText: String
    @Published var densityMaxText: String
    @Published var colorLevels: Int
    @Published var outlineWidthText: String
    @Published var useNearestFilter: Bool

    private var lastSettings: RendererSettings

    init(config: RuntimeConfig) {
        let initial = RendererSettings(
            numDistributions: config.numDistributions,
            gridResolution: config.gridResolution,
            gridMin: config.gridMin,
            gridMax: config.gridMax,
            meanStdDev: config.meanStdDev,
            covarianceScale: config.covarianceScale,
            seed: config.seed,
            planeNormal: config.planeNormal,
            colormap: config.colormap,
            invertColormap: config.invertColormap,
            useLogScale: config.useLogScale,
            densityMin: config.densityMin,
            densityMax: config.densityMax,
            colorLevels: config.colorLevels,
            outlineWidth: config.outlineWidth,
            filterMode: config.filterMode
        )
        lastSettings = initial

        numDistributionsText = String(config.numDistributions)
        gridResolutionText = String(config.gridResolution)
        gridMinText = Self.format(config.gridMin)
        gridMaxText = Self.format(config.gridMax)
        meanStdDevText = Self.format(config.meanStdDev)
        covarianceScaleText = Self.format(config.covarianceScale)
        seedString = config.seed.map(String.init) ?? ""
        planeNormalXText = Self.format(config.planeNormal.x)
        planeNormalYText = Self.format(config.planeNormal.y)
        planeNormalZText = Self.format(config.planeNormal.z)
        colormap = config.colormap
        invertColormap = config.invertColormap
        useLogScale = config.useLogScale
        densityMinText = Self.format(config.densityMin)
        densityMaxText = Self.format(config.densityMax)
        colorLevels = Int(config.colorLevels)
        outlineWidthText = Self.format(config.outlineWidth)
        useNearestFilter = config.filterMode != 0
    }

    func makeRendererSettings() -> RendererSettings {
        var sanitized = lastSettings

        if let parsed = Int(numDistributionsText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.numDistributions = max(1, parsed)
        }
        if let parsed = Int(gridResolutionText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.gridResolution = max(1, parsed)
        }
        if let parsed = Float(gridMinText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.gridMin = parsed
        }
        if let parsed = Float(gridMaxText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.gridMax = parsed
        }
        if sanitized.gridMin >= sanitized.gridMax {
            swap(&sanitized.gridMin, &sanitized.gridMax)
        }

        if let parsed = Float(meanStdDevText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.meanStdDev = max(parsed, 1e-4)
        }
        if let parsed = Float(covarianceScaleText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.covarianceScale = max(parsed, 1e-4)
        }

        let trimmedSeed = seedString.trimmingCharacters(in: .whitespacesAndNewlines)
        sanitized.seed = trimmedSeed.isEmpty ? nil : UInt64(trimmedSeed)

        let x = Float(planeNormalXText.trimmingCharacters(in: .whitespacesAndNewlines))
        let y = Float(planeNormalYText.trimmingCharacters(in: .whitespacesAndNewlines))
        let z = Float(planeNormalZText.trimmingCharacters(in: .whitespacesAndNewlines))

        if let x, let y, let z {
            sanitized.planeNormal = SIMD3<Float>(x, y, z)
        }
        if simd_length(sanitized.planeNormal) < 1e-6 {
            sanitized.planeNormal = SIMD3<Float>(1, 0, 0)
        }
        sanitized.planeNormal = simd_normalize(sanitized.planeNormal)

        let minDensityFloor: Float = 1e-16
        if let parsed = Float(densityMinText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.densityMin = max(parsed, minDensityFloor)
        }
        if let parsed = Float(densityMaxText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.densityMax = max(parsed, minDensityFloor)
        }
        if sanitized.densityMin >= sanitized.densityMax {
            swap(&sanitized.densityMin, &sanitized.densityMax)
        }

        var sanitizedColorLevels = max(0, colorLevels)
        if sanitizedColorLevels > 256 {
            sanitizedColorLevels = 256
        }
        sanitized.colorLevels = UInt32(sanitizedColorLevels)

        if let parsed = Float(outlineWidthText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            sanitized.outlineWidth = max(parsed, 0.0)
        }

        sanitized.colormap = colormap
        sanitized.invertColormap = invertColormap
        sanitized.useLogScale = useLogScale
        sanitized.filterMode = useNearestFilter ? 1 : 0

        lastSettings = sanitized
        return sanitized
    }

    func sync(from settings: RendererSettings) {
        lastSettings = settings

        numDistributionsText = String(settings.numDistributions)
        gridResolutionText = String(settings.gridResolution)
        gridMinText = Self.format(settings.gridMin)
        gridMaxText = Self.format(settings.gridMax)
        meanStdDevText = Self.format(settings.meanStdDev)
        covarianceScaleText = Self.format(settings.covarianceScale)
        seedString = settings.seed.map(String.init) ?? ""
        planeNormalXText = Self.format(settings.planeNormal.x)
        planeNormalYText = Self.format(settings.planeNormal.y)
        planeNormalZText = Self.format(settings.planeNormal.z)
        colormap = settings.colormap
        invertColormap = settings.invertColormap
        useLogScale = settings.useLogScale
        densityMinText = Self.format(settings.densityMin)
        densityMaxText = Self.format(settings.densityMax)
        colorLevels = Int(settings.colorLevels)
        outlineWidthText = Self.format(settings.outlineWidth)
        useNearestFilter = settings.filterMode != 0
    }

    private static func format<T: BinaryFloatingPoint>(_ value: T) -> String {
        String(format: "%g", Double(value))
    }
}
