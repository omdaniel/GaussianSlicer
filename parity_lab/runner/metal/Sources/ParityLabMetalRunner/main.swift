import Foundation
import Metal
import simd

struct Options {
    var casesURL: URL?
    var epsilon: Float = 1e-6
    var printLayoutOnly: Bool = false
}

enum RunnerError: Error, CustomStringConvertible {
    case invalidArgument(String)
    case invalidCaseShape(String)
    case metalUnavailable(String)
    case validationFailed(String)

    var description: String {
        switch self {
        case .invalidArgument(let arg):
            return "Unrecognized or malformed argument: \(arg)"
        case .invalidCaseShape(let reason):
            return "Invalid case data: \(reason)"
        case .metalUnavailable(let msg):
            return "Metal unavailable: \(msg)"
        case .validationFailed(let msg):
            return "Validation failed: \(msg)"
        }
    }
}

enum CaseSet {
    case scalar(label: String, cases: [ScalarVectorCase])
    case mat(label: String, cases: [Mat3Case])
    case spd(label: String, cases: [Spd3Case])

    var label: String {
        switch self {
        case .scalar(let label, _), .mat(let label, _), .spd(let label, _):
            return label
        }
    }
}

struct ScalarVectorCase {
    var alpha: Float
    var x: SIMD3<Float>
    var y: SIMD3<Float>

    func reference(epsilon: Float) -> ScalarVectorReference {
        let dot = simd_dot(x, y)
        let lengthSq = simd_length_squared(x)
        let axpy = SIMD3<Float>(
            y.x.addingProduct(alpha, x.x),
            y.y.addingProduct(alpha, x.y),
            y.z.addingProduct(alpha, x.z)
        )
        let (normalized, didNormalize) = normalizeGuarded(x, epsilon: epsilon)
        return ScalarVectorReference(dot: dot, lengthSq: lengthSq, axpy: axpy, normalized: normalized, didNormalize: didNormalize)
    }

    func gpuPayload() -> ScalarVectorCaseGpu {
        ScalarVectorCaseGpu(alpha: SIMD4<Float>(alpha, 0, 0, 0), x: SIMD4<Float>(x, 0), y: SIMD4<Float>(y, 0))
    }
}

struct ScalarVectorReference {
    var dot: Float
    var lengthSq: Float
    var axpy: SIMD3<Float>
    var normalized: SIMD3<Float>
    var didNormalize: Bool
}

struct Mat3Case {
    var a: simd_float3x3
    var b: simd_float3x3
    var v: SIMD3<Float>

    func reference() -> Mat3Reference {
        Mat3Reference(aTimesV: a * v, aTimesB: simd_mul(a, b), aTranspose: a.transpose)
    }

    func gpuPayload() -> Mat3CaseGpu {
        Mat3CaseGpu(a: Mat3ColumnsGpu(matrix: a), b: Mat3ColumnsGpu(matrix: b), v: SIMD4<Float>(v, 0))
    }
}

struct Mat3Reference {
    var aTimesV: SIMD3<Float>
    var aTimesB: simd_float3x3
    var aTranspose: simd_float3x3
}

struct Spd3Case {
    var a: simd_float3x3
    var b: SIMD3<Float>

    func reference(epsilon: Float) -> Spd3Reference {
        let chol = choleskyLower(matrix: a, epsilon: epsilon)
        let y = forwardSubstitution(l: chol, b: b)
        let x = backwardSubstitution(l: chol, y: y)
        return Spd3Reference(cholLower: chol, solution: x)
    }

    func gpuPayload() -> SpdCaseGpu {
        SpdCaseGpu(a: Mat3ColumnsGpu(matrix: a), b: SIMD4<Float>(b, 0))
    }
}

struct Spd3Reference {
    var cholLower: simd_float3x3
    var solution: SIMD3<Float>
}

struct ScalarVectorCaseGpu {
    var alpha: SIMD4<Float>
    var x: SIMD4<Float>
    var y: SIMD4<Float>
}

struct ScalarVectorResultGpu {
    var dotLen: SIMD4<Float>
    var normalized: SIMD4<Float>
    var axpy: SIMD4<Float>
    var flags: SIMD4<UInt32>

    func toReference() -> ScalarVectorReference {
        ScalarVectorReference(
            dot: dotLen.x,
            lengthSq: dotLen.y,
            axpy: SIMD3<Float>(axpy.x, axpy.y, axpy.z),
            normalized: SIMD3<Float>(normalized.x, normalized.y, normalized.z),
            didNormalize: flags.x != 0
        )
    }
}

struct Mat3ColumnsGpu {
    var c0: SIMD4<Float>
    var c1: SIMD4<Float>
    var c2: SIMD4<Float>

    init(matrix: simd_float3x3) {
        c0 = SIMD4<Float>(matrix.columns.0, 0)
        c1 = SIMD4<Float>(matrix.columns.1, 0)
        c2 = SIMD4<Float>(matrix.columns.2, 0)
    }
}

struct Mat3CaseGpu {
    var a: Mat3ColumnsGpu
    var b: Mat3ColumnsGpu
    var v: SIMD4<Float>
}

struct Mat3ResultGpu {
    var aTimesV: SIMD4<Float>
    var abCol0: SIMD4<Float>
    var abCol1: SIMD4<Float>
    var abCol2: SIMD4<Float>
    var atCol0: SIMD4<Float>
    var atCol1: SIMD4<Float>
    var atCol2: SIMD4<Float>

    func toReference() -> Mat3Reference {
        Mat3Reference(
            aTimesV: SIMD3<Float>(aTimesV.x, aTimesV.y, aTimesV.z),
            aTimesB: simd_float3x3(columns: (
                SIMD3<Float>(abCol0.x, abCol0.y, abCol0.z),
                SIMD3<Float>(abCol1.x, abCol1.y, abCol1.z),
                SIMD3<Float>(abCol2.x, abCol2.y, abCol2.z)
            )),
            aTranspose: simd_float3x3(columns: (
                SIMD3<Float>(atCol0.x, atCol0.y, atCol0.z),
                SIMD3<Float>(atCol1.x, atCol1.y, atCol1.z),
                SIMD3<Float>(atCol2.x, atCol2.y, atCol2.z)
            ))
        )
    }
}

struct SpdCaseGpu {
    var a: Mat3ColumnsGpu
    var b: SIMD4<Float>
}

struct SpdResultGpu {
    var cholCol0: SIMD4<Float>
    var cholCol1: SIMD4<Float>
    var cholCol2: SIMD4<Float>
    var solution: SIMD4<Float>

    func toReference() -> Spd3Reference {
        Spd3Reference(
            cholLower: simd_float3x3(columns: (
                SIMD3<Float>(cholCol0.x, cholCol0.y, cholCol0.z),
                SIMD3<Float>(cholCol1.x, cholCol1.y, cholCol1.z),
                SIMD3<Float>(cholCol2.x, cholCol2.y, cholCol2.z)
            )),
            solution: SIMD3<Float>(solution.x, solution.y, solution.z)
        )
    }
}

struct ConfigUniform {
    var count: UInt32
    var _pad0: UInt32
    var epsilon: Float
    var _pad1: Float
}

struct ParityLabMetalRunner {
    static func run() throws {
        let options = try parseOptions()
        let caseSets = try loadCaseSets(url: options.casesURL)

        print("Loaded \(caseSets.count) case sets")
        dumpLayout()

        if options.printLayoutOnly {
            print("Layout inspection requested. Skipping compute dispatch.")
            return
        }

        guard !caseSets.isEmpty else {
            print("No cases to run.")
            return
        }

        let harness = try MetalHarness()
        for set in caseSets {
            switch set {
            case .scalar(let label, let cases):
                try runScalarSet(label: label, cases: cases, epsilon: options.epsilon, harness: harness)
            case .mat(let label, let cases):
                try runMatSet(label: label, cases: cases, harness: harness)
            case .spd(let label, let cases):
                try runSpdSet(label: label, cases: cases, epsilon: options.epsilon, harness: harness)
            }
        }
    }
}

extension ParityLabMetalRunner {
    private static func runScalarSet(label: String, cases: [ScalarVectorCase], epsilon: Float, harness: MetalHarness) throws {
        let gpuRefs = try harness.runScalar(cases: cases, epsilon: epsilon)
        var maxDot: Float = 0
        var maxLength: Float = 0
        var maxAxpy: Float = 0
        var maxNorm: Float = 0
        var mismatched = 0
        for (cpuCase, gpuRef) in zip(cases, gpuRefs) {
            let cpuRef = cpuCase.reference(epsilon: epsilon)
            maxDot = max(maxDot, abs(cpuRef.dot - gpuRef.dot))
            maxLength = max(maxLength, abs(cpuRef.lengthSq - gpuRef.lengthSq))
            maxAxpy = max(maxAxpy, maxComponentMagnitude(cpuRef.axpy - gpuRef.axpy))
            maxNorm = max(maxNorm, maxComponentMagnitude(cpuRef.normalized - gpuRef.normalized))
            if cpuRef.didNormalize != gpuRef.didNormalize {
                mismatched += 1
            }
        }
        print("Scalar set '\(label)': max |dot|=", String(format: "%.3e", maxDot),
              " max |length²|=", String(format: "%.3e", maxLength),
              " max |axpy|=", String(format: "%.3e", maxAxpy),
              " max |norm|=", String(format: "%.3e", maxNorm),
              " flag mismatches=\(mismatched)")
        let tol: Float = 1e-6
        if maxDot > tol || maxLength > tol || maxAxpy > tol || maxNorm > tol || mismatched > 0 {
            throw RunnerError.validationFailed("Scalar set \(label) exceeded tolerance \(tol)")
        }
    }

    private static func runMatSet(label: String, cases: [Mat3Case], harness: MetalHarness) throws {
        let gpuRefs = try harness.runMat(cases: cases)
        var maxVec: Float = 0
        var maxMat: Float = 0
        for (cpuCase, gpuRef) in zip(cases, gpuRefs) {
            let cpuRef = cpuCase.reference()
            maxVec = max(maxVec, simd_length(cpuRef.aTimesV - gpuRef.aTimesV))
            maxMat = max(maxMat, maxComponentDelta(cpuRef.aTimesB, gpuRef.aTimesB))
            maxMat = max(maxMat, maxComponentDelta(cpuRef.aTranspose, gpuRef.aTranspose))
        }
        print("Mat3 set '\(label)': max |a*v|=", String(format: "%.3e", maxVec),
              " max |mat components|=", String(format: "%.3e", maxMat))
        if maxVec > 1e-6 || maxMat > 1e-6 {
            throw RunnerError.validationFailed("Mat3 set \(label) exceeded tolerance")
        }
    }

    private static func runSpdSet(label: String, cases: [Spd3Case], epsilon: Float, harness: MetalHarness) throws {
        let gpuRefs = try harness.runSpd(cases: cases, epsilon: epsilon)
        var maxChol: Float = 0
        var maxSolution: Float = 0
        for (cpuCase, gpuRef) in zip(cases, gpuRefs) {
            let residual = cpuCase.a * gpuRef.solution - cpuCase.b
            maxSolution = max(maxSolution, maxComponentMagnitude(residual))
            let cpuRef = cpuCase.reference(epsilon: epsilon)
            maxChol = max(maxChol, maxComponentDelta(cpuRef.cholLower, gpuRef.cholLower))
        }
        print("SPD set '\(label)': max |A·x - b|=", String(format: "%.3e", maxSolution),
              " max |chol|=", String(format: "%.3e", maxChol))
        if maxSolution > 1e-6 || maxChol > 1e-6 {
            throw RunnerError.validationFailed("SPD set \(label) exceeded tolerance")
        }
    }

    private static func parseOptions() throws -> Options {
        var options = Options()
        for arg in CommandLine.arguments.dropFirst() {
            if arg == "--print-layout" {
                options.printLayoutOnly = true
            } else if let value = arg.splitOnce(on: "--cases=") {
                options.casesURL = URL(fileURLWithPath: String(value))
            } else if let value = arg.splitOnce(on: "--epsilon=") {
                guard let parsed = Float(value) else {
                    throw RunnerError.invalidArgument(arg)
                }
                options.epsilon = parsed
            } else {
                throw RunnerError.invalidArgument(arg)
            }
        }
        return options
    }

    private static func loadCaseSets(url: URL?) throws -> [CaseSet] {
        if let url = url {
            let data = try Data(contentsOf: url)
            return try parseCaseSets(from: data)
        }
        return defaultCaseSets()
    }

    private static func parseCaseSets(from data: Data) throws -> [CaseSet] {
        let rootObj = try JSONSerialization.jsonObject(with: data, options: [])
        guard let array = rootObj as? [Any] else {
            throw RunnerError.invalidCaseShape("Cases JSON must be an array")
        }
        var sets: [CaseSet] = []
        var legacy: [ScalarVectorCase] = []
        for entry in array {
            if let dict = entry as? [String: Any], dict["label"] != nil || dict["cases"] != nil {
                let label = dict["label"] as? String ?? "unnamed"
                let kind = (dict["kind"] as? String) ?? "scalar_vector"
                guard let casesArray = dict["cases"] as? [Any] else {
                    throw RunnerError.invalidCaseShape("Missing cases array for set \(label)")
                }
                switch kind {
                case "scalar_vector":
                    let scalarCases = try casesArray.map { entry -> ScalarVectorCase in
                        guard let dict = entry as? [String: Any] else {
                            throw RunnerError.invalidCaseShape("Case entry in set \(label) is not an object")
                        }
                        return try parseScalarCase(dict)
                    }
                    sets.append(.scalar(label: label, cases: scalarCases))
                case "mat3_ops":
                    let matCases = try casesArray.map { entry -> Mat3Case in
                        guard let dict = entry as? [String: Any] else {
                            throw RunnerError.invalidCaseShape("Case entry in set \(label) is not an object")
                        }
                        return try parseMatCase(dict)
                    }
                    sets.append(.mat(label: label, cases: matCases))
                case "spd3":
                    let spdCases = try casesArray.map { entry -> Spd3Case in
                        guard let dict = entry as? [String: Any] else {
                            throw RunnerError.invalidCaseShape("Case entry in set \(label) is not an object")
                        }
                        return try parseSpdCase(dict)
                    }
                    sets.append(.spd(label: label, cases: spdCases))
                default:
                    throw RunnerError.invalidCaseShape("Unknown case set kind \(kind)")
                }
            } else if let dict = entry as? [String: Any], dict["alpha"] != nil {
                legacy.append(try parseScalarCase(dict))
            } else {
                throw RunnerError.invalidCaseShape("Unexpected JSON entry: \(entry)")
            }
        }
        if !legacy.isEmpty {
            sets.append(.scalar(label: "legacy", cases: legacy))
        }
        return sets
    }

    private static func parseScalarCase(_ dict: [String: Any]) throws -> ScalarVectorCase {
        guard let alpha = dict["alpha"] as? NSNumber,
              let xArr = dict["x"] as? [NSNumber], xArr.count == 3,
              let yArr = dict["y"] as? [NSNumber], yArr.count == 3 else {
            throw RunnerError.invalidCaseShape("Scalar case missing fields")
        }
        return ScalarVectorCase(alpha: alpha.floatValue,
                                x: SIMD3<Float>(xArr[0].floatValue, xArr[1].floatValue, xArr[2].floatValue),
                                y: SIMD3<Float>(yArr[0].floatValue, yArr[1].floatValue, yArr[2].floatValue))
    }

    private static func parseMatCase(_ dict: [String: Any]) throws -> Mat3Case {
        guard let aRows = dict["a"] as? [[NSNumber]], aRows.count == 3,
              let bRows = dict["b"] as? [[NSNumber]], bRows.count == 3,
              let vArr = dict["v"] as? [NSNumber], vArr.count == 3 else {
            throw RunnerError.invalidCaseShape("Mat3 case missing fields")
        }
        let a = matrixFromRows(aRows)
        let b = matrixFromRows(bRows)
        let v = SIMD3<Float>(vArr[0].floatValue, vArr[1].floatValue, vArr[2].floatValue)
        return Mat3Case(a: a, b: b, v: v)
    }

    private static func parseSpdCase(_ dict: [String: Any]) throws -> Spd3Case {
        guard let aRows = dict["a"] as? [[NSNumber]], aRows.count == 3,
              let bArr = dict["b"] as? [NSNumber], bArr.count == 3 else {
            throw RunnerError.invalidCaseShape("SPD case missing fields")
        }
        let a = matrixFromRows(aRows)
        let b = SIMD3<Float>(bArr[0].floatValue, bArr[1].floatValue, bArr[2].floatValue)
        return Spd3Case(a: a, b: b)
    }

    private static func matrixFromRows(_ rows: [[NSNumber]]) -> simd_float3x3 {
        func rowVec(_ row: [NSNumber]) -> SIMD3<Float> {
            SIMD3<Float>(row[0].floatValue, row[1].floatValue, row[2].floatValue)
        }
        let r0 = rowVec(rows[0])
        let r1 = rowVec(rows[1])
        let r2 = rowVec(rows[2])
        return simd_float3x3(columns: (
            SIMD3<Float>(r0.x, r1.x, r2.x),
            SIMD3<Float>(r0.y, r1.y, r2.y),
            SIMD3<Float>(r0.z, r1.z, r2.z)
        ))
    }


    private static func defaultCaseSets() -> [CaseSet] {
        let scalarDefaults = [
            ScalarVectorCase(alpha: 1.0, x: SIMD3<Float>(1, 0, 0), y: SIMD3<Float>(0, 1, 0)),
            ScalarVectorCase(alpha: 0.5, x: SIMD3<Float>(-1, 2, -3), y: SIMD3<Float>(4, -5, 6)),
            ScalarVectorCase(alpha: -2.0, x: SIMD3<Float>(0.1, 0.2, 0.3), y: SIMD3<Float>(0.3, 0.2, 0.1))
        ]
        let matDefaults = [
            Mat3Case(a: matrixFromSimdRows(rows: (SIMD3<Float>(1, 2, 0), SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1))),
                     b: matrixFromSimdRows(rows: (SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1))),
                     v: SIMD3<Float>(1, 0, 0)),
            Mat3Case(a: matrixFromSimdRows(rows: (SIMD3<Float>(0.5, -1, 0), SIMD3<Float>(1, 0.5, 0.25), SIMD3<Float>(0, 0.25, 0.75))),
                     b: matrixFromSimdRows(rows: (SIMD3<Float>(0.1, 0.2, 0.3), SIMD3<Float>(0.4, 0.5, 0.6), SIMD3<Float>(0.7, 0.8, 0.9))),
                     v: SIMD3<Float>(-1, 2, -3))
        ]
        let spdDefaults = [
            Spd3Case(a: matrixFromSimdRows(rows: (SIMD3<Float>(2, 0, 0), SIMD3<Float>(0, 3, 0), SIMD3<Float>(0, 0, 4))),
                     b: SIMD3<Float>(1, 0, 0)),
            Spd3Case(a: matrixFromSimdRows(rows: (SIMD3<Float>(1.5, 0.2, 0.1), SIMD3<Float>(0.2, 2.0, 0.3), SIMD3<Float>(0.1, 0.3, 2.5))),
                     b: SIMD3<Float>(-0.5, 1.0, 0.25))
        ]
        return [
            .scalar(label: "default_scalar", cases: scalarDefaults),
            .mat(label: "default_mat", cases: matDefaults),
            .spd(label: "default_spd", cases: spdDefaults)
        ]
    }

    private static func dumpLayout() {
        func describe<T>(_ type: T.Type) {
            print("\(type) stride=\(MemoryLayout<T>.stride), alignment=\(MemoryLayout<T>.alignment)")
        }
        describe(ScalarVectorCaseGpu.self)
        describe(ScalarVectorResultGpu.self)
        describe(Mat3CaseGpu.self)
        describe(Mat3ResultGpu.self)
        describe(SpdCaseGpu.self)
        describe(SpdResultGpu.self)
        describe(ConfigUniform.self)
    }
}

struct MetalHarness {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let scalarPipeline: MTLComputePipelineState
    let matPipeline: MTLComputePipelineState
    let spdPipeline: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RunnerError.metalUnavailable("no compatible device found")
        }
        guard let queue = device.makeCommandQueue() else {
            throw RunnerError.metalUnavailable("failed to create command queue")
        }
        let library: MTLLibrary
        if let bundleLibrary = try? device.makeDefaultLibrary(bundle: .module) {
            library = bundleLibrary
        } else if let sourceURL = Bundle.module.url(forResource: "Shaders", withExtension: "metal") {
            let source = try String(contentsOf: sourceURL)
            library = try device.makeLibrary(source: source, options: nil)
        } else if let fallback = device.makeDefaultLibrary() {
            library = fallback
        } else {
            throw RunnerError.metalUnavailable("failed to load Metal library")
        }
        guard
            let scalarFn = library.makeFunction(name: "scalarVectorKernel"),
            let matFn = library.makeFunction(name: "mat3Kernel"),
            let spdFn = library.makeFunction(name: "spd3Kernel")
        else {
            throw RunnerError.metalUnavailable("missing shader functions")
        }
        self.scalarPipeline = try device.makeComputePipelineState(function: scalarFn)
        self.matPipeline = try device.makeComputePipelineState(function: matFn)
        self.spdPipeline = try device.makeComputePipelineState(function: spdFn)
        self.device = device
        self.queue = queue
    }

    func runScalar(cases: [ScalarVectorCase], epsilon: Float) throws -> [ScalarVectorReference] {
        guard !cases.isEmpty else { return [] }
        let payload = cases.map { $0.gpuPayload() }
        let resultBuffer = try dispatch(payload: payload, resultStride: MemoryLayout<ScalarVectorResultGpu>.stride, pipeline: scalarPipeline, epsilon: epsilon)
        let pointer = resultBuffer.contents().bindMemory(to: ScalarVectorResultGpu.self, capacity: payload.count)
        return (0..<payload.count).map { pointer[$0].toReference() }
    }

    func runMat(cases: [Mat3Case]) throws -> [Mat3Reference] {
        guard !cases.isEmpty else { return [] }
        let payload = cases.map { $0.gpuPayload() }
        let resultBuffer = try dispatch(payload: payload, resultStride: MemoryLayout<Mat3ResultGpu>.stride, pipeline: matPipeline, epsilon: 0)
        let pointer = resultBuffer.contents().bindMemory(to: Mat3ResultGpu.self, capacity: payload.count)
        return (0..<payload.count).map { pointer[$0].toReference() }
    }

    func runSpd(cases: [Spd3Case], epsilon: Float) throws -> [Spd3Reference] {
        guard !cases.isEmpty else { return [] }
        let payload = cases.map { $0.gpuPayload() }
        let resultBuffer = try dispatch(payload: payload, resultStride: MemoryLayout<SpdResultGpu>.stride, pipeline: spdPipeline, epsilon: epsilon)
        let pointer = resultBuffer.contents().bindMemory(to: SpdResultGpu.self, capacity: payload.count)
        return (0..<payload.count).map { pointer[$0].toReference() }
    }

    private func dispatch<Payload>(payload: [Payload], resultStride: Int, pipeline: MTLComputePipelineState, epsilon: Float) throws -> MTLBuffer {
        let payloadBuffer = try makeBuffer(bytes: payload)
        guard let resultBuffer = device.makeBuffer(length: resultStride * payload.count, options: .storageModeShared) else {
            throw RunnerError.metalUnavailable("failed to allocate result buffer")
        }
        var config = ConfigUniform(count: UInt32(payload.count), _pad0: 0, epsilon: epsilon, _pad1: 0)
        guard let configBuffer = device.makeBuffer(bytes: &config, length: MemoryLayout<ConfigUniform>.stride, options: .storageModeShared) else {
            throw RunnerError.metalUnavailable("failed to allocate config buffer")
        }
        guard let commandBuffer = queue.makeCommandBuffer(), let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RunnerError.metalUnavailable("failed to create command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(payloadBuffer, offset: 0, index: 0)
        encoder.setBuffer(resultBuffer, offset: 0, index: 1)
        encoder.setBuffer(configBuffer, offset: 0, index: 2)
        let threadsPerGroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        let threads = MTLSize(width: payload.count, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return resultBuffer
    }

    private func makeBuffer<T>(bytes: [T]) throws -> MTLBuffer {
        let length = bytes.count * MemoryLayout<T>.stride
        if length == 0 {
            guard let buffer = device.makeBuffer(length: 0, options: .storageModeShared) else {
                throw RunnerError.metalUnavailable("failed to allocate empty buffer")
            }
            return buffer
        }
        return try bytes.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                throw RunnerError.metalUnavailable("invalid buffer data")
            }
            guard let buffer = device.makeBuffer(bytes: baseAddress, length: length, options: .storageModeShared) else {
                throw RunnerError.metalUnavailable("failed to allocate buffer")
            }
            return buffer
        }
    }
}

private func normalizeGuarded(_ v: SIMD3<Float>, epsilon: Float) -> (SIMD3<Float>, Bool) {
    let lenSq = simd_length_squared(v)
    if lenSq <= epsilon {
        return (v, false)
    }
    let inv = 1.0 / sqrt(lenSq)
    return (v * inv, true)
}

private func maxComponentDelta(_ lhs: simd_float3x3, _ rhs: simd_float3x3) -> Float {
    var delta: Float = 0
    for i in 0..<3 {
        for j in 0..<3 {
            delta = max(delta, abs(lhs[i, j] - rhs[i, j]))
        }
    }
    return delta
}

private func maxComponentMagnitude(_ v: SIMD3<Float>) -> Float {
    max(abs(v.x), max(abs(v.y), abs(v.z)))
}

private func choleskyLower(matrix: simd_float3x3, epsilon: Float) -> simd_float3x3 {
    var L = matrixFromSimdRows(rows: (SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0)))
    for i in 0..<3 {
        for j in 0...i {
            var sum = matrix[i, j]
            for k in 0..<j {
                sum -= L[i, k] * L[j, k]
            }
            if i == j {
                L[i, j] = sqrt(max(sum, epsilon))
            } else {
                L[i, j] = sum / L[j, j]
            }
        }
    }
    return L
}

private func forwardSubstitution(l: simd_float3x3, b: SIMD3<Float>) -> SIMD3<Float> {
    let y0 = b.x / l[0,0]
    let y1 = (b.y - l[1,0] * y0) / l[1,1]
    let y2 = (b.z - l[2,0] * y0 - l[2,1] * y1) / l[2,2]
    return SIMD3<Float>(y0, y1, y2)
}

private func backwardSubstitution(l: simd_float3x3, y: SIMD3<Float>) -> SIMD3<Float> {
    let x2 = y.z / l[2,2]
    let x1 = (y.y - l[2,1] * x2) / l[1,1]
    let x0 = (y.x - l[1,0] * x1 - l[2,0] * x2) / l[0,0]
    return SIMD3<Float>(x0, x1, x2)
}

private extension String {
    func splitOnce(on prefix: String) -> Substring? {
        guard self.hasPrefix(prefix) else { return nil }
        let idx = self.index(self.startIndex, offsetBy: prefix.count)
        return self[idx...]
    }
}

do {
    try ParityLabMetalRunner.run()
} catch {
    fputs("ParityLabMetalRunner error: \(error)\n", stderr)
    exit(EXIT_FAILURE)
}
private func matrixFromSimdRows(rows: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)) -> simd_float3x3 {
    let (r0, r1, r2) = rows
    return simd_float3x3(columns: (
        SIMD3<Float>(r0.x, r1.x, r2.x),
        SIMD3<Float>(r0.y, r1.y, r2.y),
        SIMD3<Float>(r0.z, r1.z, r2.z)
    ))
}
