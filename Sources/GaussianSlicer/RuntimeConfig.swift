import Foundation
import simd

enum ColormapOption: UInt32, CaseIterable {
    case plasma = 0
    case viridis
    case magma
    case inferno
    case turbo
    case coolWarm
    case blueOrange
    case seismic
    case ylOrRd
    case hot
}

struct RuntimeConfig {
    static let shared = RuntimeConfig()

    let numDistributions: Int
    let gridResolution: Int
    let gridMin: Float
    let gridMax: Float
    let meanStdDev: Float
    let covarianceScale: Float
    let seed: UInt64?
    let planeNormal: SIMD3<Float>
    let colormap: ColormapOption
    let invertColormap: Bool
    let useLogScale: Bool
    let densityMin: Float
    let densityMax: Float
    let colorLevels: UInt32
    let outlineWidth: Float
    let shouldExit: Bool
    let gaussianPlyURL: URL?
    let captureFrameURL: URL?
    let exportVolumeURL: URL?
    let exportLogNormalized: Bool
    let filterMode: UInt32

    private init() {
        let defaultPlaneNormal = SIMD3<Float>(1.0, 0.5, 0.8)
        var numDistributions = 50_000
        var gridResolution = 256
        var gridMin: Float = -8.0
        var gridMax: Float = 8.0
        var meanStdDev: Float = 2.5
        var covarianceScale: Float = 0.1
        var seed: UInt64? = nil
        var planeNormal = defaultPlaneNormal
        var colormap: ColormapOption = .plasma
        var invertColormap = false
        var useLogScale = true
        var densityMin: Float = 1e-6
        var densityMax: Float = 0.05
        var colorLevels: UInt32 = 0
        var outlineWidth: Float = 0.0
        var shouldExit = false
        var gaussianPlyURL: URL? = nil
        var captureFrameURL: URL? = nil
        var exportVolumeURL: URL? = nil
        var exportLogNormalized = false
        var filterMode: UInt32 = 0

        for argument in CommandLine.arguments.dropFirst() {
            guard argument.hasPrefix("--") else { continue }
            let parts = argument.dropFirst(2).split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            let key = parts.first?.lowercased() ?? ""
            if key.isEmpty {
                continue
            }
            let value = parts.count > 1 ? String(parts[1]) : nil

            switch key {
            case "help":
                RuntimeConfig.printHelp()
                shouldExit = true
            case "num-distributions":
                if let value, let parsed = Int(value), parsed > 0 {
                    numDistributions = parsed
                }
            case "grid-resolution":
                if let value, let parsed = Int(value), parsed > 0 {
                    gridResolution = parsed
                }
            case "grid-min":
                if let value, let parsed = Float(value) {
                    gridMin = parsed
                }
            case "grid-max":
                if let value, let parsed = Float(value) {
                    gridMax = parsed
                }
            case "mean-stddev":
                if let value, let parsed = Float(value), parsed > 0 {
                    meanStdDev = parsed
                }
            case "covariance-scale":
                if let value, let parsed = Float(value), parsed > 0 {
                    covarianceScale = parsed
                }
            case "seed":
                if let value, let parsed = UInt64(value) {
                    seed = parsed
                }
            case "plane-normal":
                if let value {
                    let components = value.split(whereSeparator: { $0 == "," || $0 == ":" })
                    if components.count == 3,
                       let x = Float(components[0]),
                       let y = Float(components[1]),
                       let z = Float(components[2]) {
                        planeNormal = SIMD3<Float>(x, y, z)
                    }
                }
            case "colormap":
                if let value {
                    let token = value.lowercased()
                    if let option = ColormapOption.allCases.first(where: { String(describing: $0).lowercased() == token }) {
                        colormap = option
                    } else {
                        print("Warning: Unsupported colormap '\(value)'. Using default 'plasma'.")
                    }
                }
            case "invert-colormap":
                if let value {
                    if let parsed = RuntimeConfig.parseBool(from: value) {
                        invertColormap = parsed
                    }
                } else {
                    invertColormap = true
                }
           case "log-scale":
               if let value {
                   if let parsed = RuntimeConfig.parseBool(from: value) {
                       useLogScale = parsed
                   }
               } else {
                   useLogScale.toggle()
               }
            case "density-min":
                if let value, let parsed = Float(value), parsed > 0 {
                    densityMin = parsed
                }
            case "density-max":
                if let value, let parsed = Float(value), parsed > 0 {
                    densityMax = parsed
                }
            case "color-levels", "colormap-levels":
                if let value, let parsed = Int(value), parsed >= 0 {
                    colorLevels = UInt32(parsed)
                }
            case "outline-width":
                if let value, let parsed = Float(value), parsed >= 0 {
                    outlineWidth = parsed
                }
            case "gaussian-ply":
                if let value {
                    gaussianPlyURL = URL(fileURLWithPath: value).standardizedFileURL
                }
            case "capture-frame":
                if let value {
                    captureFrameURL = URL(fileURLWithPath: value).standardizedFileURL
                }
            case "filter-mode":
                if let value {
                    let token = value.lowercased()
                    if token == "nearest" || token == "1" {
                        filterMode = 1
                    } else {
                        filterMode = 0
                    }
                }
            case "export-volume":
                if let value {
                    exportVolumeURL = URL(fileURLWithPath: value).standardizedFileURL
                }
            case "export-log-normalized":
                if let value {
                    if let parsed = RuntimeConfig.parseBool(from: value) {
                        exportLogNormalized = parsed
                    }
                } else {
                    exportLogNormalized = true
                }
            default:
                print("Warning: Unrecognized option '--\(key)'. Use --help for available flags.")
            }
        }

        if gridMin >= gridMax {
            swap(&gridMin, &gridMax)
        }

        if densityMin >= densityMax {
            swap(&densityMin, &densityMax)
        }

        if colorLevels > 256 {
            colorLevels = 256
        }

        if length(planeNormal) < 1e-6 {
            planeNormal = defaultPlaneNormal
        }

        self.numDistributions = numDistributions
        self.gridResolution = gridResolution
        self.gridMin = gridMin
        self.gridMax = gridMax
        self.meanStdDev = meanStdDev
        self.covarianceScale = covarianceScale
        self.seed = seed
        self.colormap = colormap
        self.invertColormap = invertColormap
        self.useLogScale = useLogScale
        self.densityMin = densityMin
        self.densityMax = densityMax
        self.colorLevels = colorLevels
        self.outlineWidth = outlineWidth
        self.shouldExit = shouldExit
        self.planeNormal = simd_normalize(planeNormal)
        self.gaussianPlyURL = gaussianPlyURL
        self.captureFrameURL = captureFrameURL
        self.exportVolumeURL = exportVolumeURL
        self.exportLogNormalized = exportLogNormalized
        self.filterMode = filterMode
    }

    private static func printHelp() {
        let message = """
        GaussianSlicer launch options:
          --help                      Show this message and exit.
          --num-distributions=INT     Number of mixture components (default: 50000).
          --grid-resolution=INT       Texture resolution per axis (default: 256).
          --grid-min=FLOAT            Minimum grid coordinate (default: -8).
          --grid-max=FLOAT            Maximum grid coordinate (default: 8).
          --mean-stddev=FLOAT         Standard deviation for component means (default: 2.5).
          --covariance-scale=FLOAT    Multiplier applied to SPD covariances (default: 0.1).
          --seed=UINT64               Use a fixed RNG seed for reproducibility.
          --plane-normal=x,y,z        Override slice plane normal before normalization.
          --colormap=NAME             One of plasma, viridis, magma, inferno, turbo, coolwarm, blueorange, seismic, ylorrd, hot.
          --invert-colormap[=BOOL]    Flip the gradient direction (default: false).
          --log-scale[=BOOL]          Use logarithmic density scaling (default: true).
          --density-min=FLOAT         Minimum density for color mapping (default: 1e-6).
          --density-max=FLOAT         Maximum density for color mapping (default: 0.05).
          --color-levels=INT          Number of discrete color bands (0 = continuous).
          --outline-width=FLOAT       Width of band outlines in points (0 = off).
          --filter-mode=MODE          bilinear (default) or nearest.
          --gaussian-ply=PATH         Load Gaussian mixture data from a Gaussian splat PLY file.
          --capture-frame=PATH        Render the visualization to a PNG at grid resolution and exit.
        """
        print(message)
    }

    private static func parseBool(from value: String) -> Bool? {
        switch value.lowercased() {
        case "1", "true", "yes", "y", "on":
            return true
        case "0", "false", "no", "n", "off":
            return false
        default:
            print("Warning: Unable to parse boolean value '\(value)'.")
            return nil
        }
    }
}
