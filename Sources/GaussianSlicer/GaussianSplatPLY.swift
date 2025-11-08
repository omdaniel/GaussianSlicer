import Foundation
import simd

enum GaussianSplatPLYError: Error {
    case malformed(String)
}

struct GaussianSplatPLY {
    static let minVariance: Float = 1e-6

    static func load(url: URL) throws -> [Gaussian3D] {
        let data = try Data(contentsOf: url)
        let (headerString, payloadOffset) = try splitHeader(from: data)
        let header = try PlyHeader.parse(from: headerString)
        let payload = data[payloadOffset..<data.endIndex]

        let gaussians: [Gaussian3D]
        switch header.format {
        case .ascii:
            let payloadString = String(decoding: payload, as: UTF8.self)
            gaussians = try parseAsciiVertices(payloadString, header: header)
        case .binaryLittleEndian:
            gaussians = try parseBinaryVertices(Data(payload), header: header)
        }

        return normalizeWeights(gaussians)
    }

    private static func splitHeader(from data: Data) throws -> (String, Data.Index) {
        guard let markerRange = data.range(of: Data("end_header".utf8)) else {
            throw GaussianSplatPLYError.malformed("PLY file is missing 'end_header'")
        }

        var payloadStart = markerRange.upperBound
        while payloadStart < data.endIndex {
            let byte = data[payloadStart]
            if byte == 0x0A { // \n
                payloadStart = data.index(after: payloadStart)
                break
            } else if byte == 0x0D { // \r
                payloadStart = data.index(after: payloadStart)
                if payloadStart < data.endIndex && data[payloadStart] == 0x0A {
                    payloadStart = data.index(after: payloadStart)
                }
                break
            } else if byte == 0x20 || byte == 0x09 { // whitespace
                payloadStart = data.index(after: payloadStart)
            } else {
                break
            }
        }

        guard let headerString = String(data: data[..<payloadStart], encoding: .ascii) else {
            throw GaussianSplatPLYError.malformed("PLY header must be ASCII encoded")
        }
        return (headerString, payloadStart)
    }

    private static func parseAsciiVertices(_ payload: String, header: PlyHeader) throws -> [Gaussian3D] {
        var gaussians: [Gaussian3D] = []
        gaussians.reserveCapacity(header.vertexCount)

        for line in payload.split(maxSplits: Int.max, omittingEmptySubsequences: true, whereSeparator: { $0.isNewline }) {
            if gaussians.count >= header.vertexCount {
                break
            }
            let tokens = line.split(maxSplits: Int.max, omittingEmptySubsequences: true, whereSeparator: { $0 == " " || $0 == "\t" })
            guard tokens.count >= header.properties.count else {
                throw GaussianSplatPLYError.malformed("ASCII vertex has fewer scalars than declared")
            }

            var builder = GaussianBuilder()
            for (property, token) in zip(header.properties, tokens) {
                let value = try property.type.parseAscii(token)
                builder.consume(name: property.name, value: Float(value))
            }
            gaussians.append(try builder.build())
        }

        guard gaussians.count == header.vertexCount else {
            throw GaussianSplatPLYError.malformed("Expected \(header.vertexCount) vertices, parsed \(gaussians.count)")
        }
        return gaussians
    }

    private static func parseBinaryVertices(_ payload: Data, header: PlyHeader) throws -> [Gaussian3D] {
        let stride = header.properties.reduce(0) { $0 + $1.type.byteSize }
        guard payload.count >= stride * header.vertexCount else {
            throw GaussianSplatPLYError.malformed("Binary payload shorter than expected")
        }

        var gaussians: [Gaussian3D] = []
        gaussians.reserveCapacity(header.vertexCount)

        var offset = 0
        for _ in 0..<header.vertexCount {
            var builder = GaussianBuilder()
            for property in header.properties {
                let value = try property.type.readLittleEndian(from: payload, offset: &offset)
                builder.consume(name: property.name, value: Float(value))
            }
            gaussians.append(try builder.build())
        }

        return gaussians
    }

    private static func normalizeWeights(_ gaussians: [Gaussian3D]) -> [Gaussian3D] {
        guard !gaussians.isEmpty else { return gaussians }
        var result = gaussians
        let total = result.reduce(0.0) { $0 + $1.weight }
        if total > 0 {
            for idx in result.indices {
                result[idx].weight /= total
            }
        } else {
            let uniform = 1.0 / Float(result.count)
            for idx in result.indices {
                result[idx].weight = uniform
            }
        }
        return result
    }
}

private struct PlyHeader {
    enum Format {
        case ascii
        case binaryLittleEndian
    }

    struct Property {
        let name: String
        let type: PlyScalarType
    }

    let format: Format
    let vertexCount: Int
    let properties: [Property]

    static func parse(from header: String) throws -> PlyHeader {
        let lines = header.split(maxSplits: Int.max, omittingEmptySubsequences: false, whereSeparator: { $0.isNewline })
        let whitespaceSet = CharacterSet.whitespacesAndNewlines
        guard let firstLine = lines.first, firstLine.trimmingCharacters(in: whitespaceSet).lowercased() == "ply" else {
            throw GaussianSplatPLYError.malformed("PLY header must start with 'ply'")
        }

        var format: Format?
        var vertexCount: Int?
        var properties: [Property] = []
        var parsingVertexElement = false

        for rawLine in lines.dropFirst() {
            let line = rawLine.trimmingCharacters(in: whitespaceSet)
            if line.isEmpty || line.hasPrefix("comment") || line.hasPrefix("obj_info") {
                continue
            }

            let tokens = line.split(maxSplits: Int.max, omittingEmptySubsequences: true, whereSeparator: { $0 == " " || $0 == "\t" })
            guard let keyword = tokens.first else { continue }

            switch keyword {
            case "format":
                guard tokens.count >= 2 else {
                    throw GaussianSplatPLYError.malformed("Invalid format declaration")
                }
                switch tokens[1] {
                case "ascii":
                    format = .ascii
                case "binary_little_endian":
                    format = .binaryLittleEndian
                default:
                    throw GaussianSplatPLYError.malformed("Unsupported PLY format '\(tokens[1])'")
                }
            case "element":
                guard tokens.count >= 3 else {
                    throw GaussianSplatPLYError.malformed("Malformed element declaration")
                }
                let name = tokens[1]
                guard let count = Int(tokens[2]) else {
                    throw GaussianSplatPLYError.malformed("Invalid element count for \(name)")
                }
                if name == "vertex" {
                    vertexCount = count
                    properties.removeAll(keepingCapacity: true)
                    parsingVertexElement = true
                } else {
                    parsingVertexElement = false
                }
            case "property":
                guard parsingVertexElement else { continue }
                guard tokens.count >= 3 else {
                    throw GaussianSplatPLYError.malformed("Malformed property declaration")
                }
                if tokens[1] == "list" {
                    throw GaussianSplatPLYError.malformed("List properties are not supported for vertex data")
                }
                let scalar = try PlyScalarType.parse(tokens[1])
                properties.append(Property(name: String(tokens[2]), type: scalar))
            case "end_header":
                break
            default:
                continue
            }
        }

        guard let parsedFormat = format else {
            throw GaussianSplatPLYError.malformed("PLY header missing format declaration")
        }
        guard let count = vertexCount else {
            throw GaussianSplatPLYError.malformed("PLY header missing vertex element declaration")
        }
        guard !properties.isEmpty else {
            throw GaussianSplatPLYError.malformed("Vertex element does not declare any properties")
        }

        return PlyHeader(format: parsedFormat, vertexCount: count, properties: properties)
    }
}

private enum PlyScalarType {
    case char
    case uchar
    case short
    case ushort
    case int
    case uint
    case float
    case double

    static func parse(_ token: Substring) throws -> PlyScalarType {
        switch token {
        case "char", "int8":
            return .char
        case "uchar", "uint8":
            return .uchar
        case "short", "int16":
            return .short
        case "ushort", "uint16":
            return .ushort
        case "int", "int32":
            return .int
        case "uint", "uint32":
            return .uint
        case "float", "float32":
            return .float
        case "double", "float64":
            return .double
        default:
            throw GaussianSplatPLYError.malformed("Unsupported scalar type '\(token)'")
        }
    }

    var byteSize: Int {
        switch self {
        case .char, .uchar:
            return 1
        case .short, .ushort:
            return 2
        case .int, .uint, .float:
            return 4
        case .double:
            return 8
        }
    }

    func parseAscii(_ token: Substring) throws -> Double {
        switch self {
        case .char:
            guard let value = Int8(token) else { throw GaussianSplatPLYError.malformed("Invalid char scalar") }
            return Double(value)
        case .uchar:
            guard let value = UInt8(token) else { throw GaussianSplatPLYError.malformed("Invalid uchar scalar") }
            return Double(value)
        case .short:
            guard let value = Int16(token) else { throw GaussianSplatPLYError.malformed("Invalid short scalar") }
            return Double(value)
        case .ushort:
            guard let value = UInt16(token) else { throw GaussianSplatPLYError.malformed("Invalid ushort scalar") }
            return Double(value)
        case .int:
            guard let value = Int32(token) else { throw GaussianSplatPLYError.malformed("Invalid int scalar") }
            return Double(value)
        case .uint:
            guard let value = UInt32(token) else { throw GaussianSplatPLYError.malformed("Invalid uint scalar") }
            return Double(value)
        case .float:
            guard let value = Float(token) else { throw GaussianSplatPLYError.malformed("Invalid float scalar") }
            return Double(value)
        case .double:
            guard let value = Double(token) else { throw GaussianSplatPLYError.malformed("Invalid double scalar") }
            return value
        }
    }

    func readLittleEndian(from data: Data, offset: inout Int) throws -> Double {
        let size = byteSize
        guard offset + size <= data.count else {
            throw GaussianSplatPLYError.malformed("Unexpected end of binary data")
        }
        let range = offset..<(offset + size)
        var buffer = [UInt8](repeating: 0, count: size)
        data.copyBytes(to: &buffer, from: range)
        offset += size

        switch self {
        case .char:
            return Double(Int8(bitPattern: buffer[0]))
        case .uchar:
            return Double(buffer[0])
        case .short:
            let raw = UInt16(buffer[0]) | UInt16(buffer[1]) << 8
            return Double(Int16(bitPattern: raw))
        case .ushort:
            let raw = UInt16(buffer[0]) | UInt16(buffer[1]) << 8
            return Double(raw)
        case .int:
            let raw = UInt32(buffer[0])
                | (UInt32(buffer[1]) << 8)
                | (UInt32(buffer[2]) << 16)
                | (UInt32(buffer[3]) << 24)
            return Double(Int32(bitPattern: raw))
        case .uint:
            let raw = UInt32(buffer[0])
                | (UInt32(buffer[1]) << 8)
                | (UInt32(buffer[2]) << 16)
                | (UInt32(buffer[3]) << 24)
            return Double(raw)
        case .float:
            let raw = UInt32(buffer[0])
                | (UInt32(buffer[1]) << 8)
                | (UInt32(buffer[2]) << 16)
                | (UInt32(buffer[3]) << 24)
            return Double(Float(bitPattern: raw))
        case .double:
            var value: UInt64 = 0
            for (index, byte) in buffer.enumerated() {
                value |= UInt64(byte) << (8 * index)
            }
            return Double(bitPattern: value)
        }
    }
}

private struct GaussianBuilder {
    var mean = SIMD3<Float>(repeating: 0)
    var scales = SIMD3<Float>(repeating: 0)
    var rotation = SIMD4<Float>(repeating: 0)
    var meanMask = [false, false, false]
    var scaleMask = [false, false, false]
    var rotMask = [false, false, false, false]
    var weight: Float?

    mutating func consume(name: String, value: Float) {
        switch name {
        case "x":
            mean.x = value
            meanMask[0] = true
        case "y":
            mean.y = value
            meanMask[1] = true
        case "z":
            mean.z = value
            meanMask[2] = true
        case "scale_0", "scale_x":
            scales.x = value
            scaleMask[0] = true
        case "scale_1", "scale_y":
            scales.y = value
            scaleMask[1] = true
        case "scale_2", "scale_z":
            scales.z = value
            scaleMask[2] = true
        case "rot_0", "rot_x":
            rotation.x = value
            rotMask[0] = true
        case "rot_1", "rot_y":
            rotation.y = value
            rotMask[1] = true
        case "rot_2", "rot_z":
            rotation.z = value
            rotMask[2] = true
        case "rot_3", "rot_w":
            rotation.w = value
            rotMask[3] = true
        case "opacity", "alpha", "weight", "w":
            weight = value
        default:
            break
        }
    }

    func build() throws -> Gaussian3D {
        guard meanMask.allSatisfy({ $0 }) else {
            throw GaussianSplatPLYError.malformed("Missing mean components in vertex record")
        }
        guard scaleMask.allSatisfy({ $0 }) else {
            throw GaussianSplatPLYError.malformed("Missing scale components in vertex record")
        }
        guard rotMask.allSatisfy({ $0 }) else {
            throw GaussianSplatPLYError.malformed("Missing rotation components in vertex record")
        }

        let rawWeight = max(weight ?? 1.0, 0.0)
        let sigma = simd_float3(
            clampedExp(scales.x),
            clampedExp(scales.y),
            clampedExp(scales.z)
        )
        var variances = sigma * sigma
        variances.x = max(variances.x, GaussianSplatPLY.minVariance)
        variances.y = max(variances.y, GaussianSplatPLY.minVariance)
        variances.z = max(variances.z, GaussianSplatPLY.minVariance)

        var quat = simd_quatf(ix: rotation.x, iy: rotation.y, iz: rotation.z, r: rotation.w)
        if simd_length(quat.imag) < 1e-6 && abs(quat.real) < 1e-6 {
            quat = simd_quatf(angle: 0, axis: SIMD3<Float>(1, 0, 0))
        } else {
            quat = simd_normalize(quat)
        }
        let rotationMatrix = simd_float3x3(quat)
        let diag = simd_float3x3(
            SIMD3<Float>(variances.x, 0, 0),
            SIMD3<Float>(0, variances.y, 0),
            SIMD3<Float>(0, 0, variances.z)
        )
        let covariance = rotationMatrix * diag * rotationMatrix.transpose
        return Gaussian3D(mean: mean, covariance: covariance, weight: rawWeight)
    }
}

private func clampedExp(_ value: Float) -> Float {
    let clamped = max(min(value, 16.0), -16.0)
    return expf(clamped)
}
