//
//  KernelDataTypes.swift
//  Warp
//
//  Created by Richard Wei on 11/2/16.
//
//

import CUDARuntime

/// Kernel operators

public extension KernelArgument {

    public static func pointer<T: DeviceAddressable>(to value: inout T) -> KernelArgument {
        return value.withUnsafeMutableDevicePointer { ptr in
            self.init(ptr.deviceAddress)
        }
    }

    public static func constPointer<T: DeviceAddressable>(to value: T) -> KernelArgument {
        return value.withUnsafeDevicePointer { ptr in
            self.init(ptr.deviceAddress)
        }
    }

    public static func value<T: KernelDataProtocol>(_ value: T) -> KernelArgument {
        return self.init(value)
    }

}

public enum KernelDataType : String {
    case float            = "float"              // 32
    case double           = "double"             // 64
    case char             = "char"               // 8
    case short            = "short"              // 16
    case int              = "int"                // 32
    case longLong         = "long long"          // 64
    case unsignedChar     = "unsigned char"      // 8
    case unsignedShort    = "unsigned short"     // 16
    case unsignedInt      = "unsigned int"       // 32
    case unsignedLongLong = "unsigned long long" // 64
}

public protocol KernelDataProtocol : ExpressibleByIntegerLiteral {
    static var kernelDataType: KernelDataType { get }
}

extension Int : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .longLong
    }
}

extension UInt : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .unsignedLongLong
    }
}

extension Float : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .float
    }
}

extension Double : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .double
    }
}

extension Int8 : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .char
    }
}

extension Int16 : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .short
    }
}

extension Int32 : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .int
    }
}

extension Int64 : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .longLong
    }
}

extension UInt8 : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .unsignedChar
    }
}

extension UInt16 : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .unsignedShort
    }
}

extension UInt32 : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .unsignedInt
    }
}

extension UInt64 : KernelDataProtocol {
    public static var kernelDataType: KernelDataType {
        return .unsignedLongLong
    }
}

public enum DeviceUnaryTransformation {
    case exp, log, cos, sin, tan, tanh, sinh, cosh, acos, asin, atan, floor, ceil
}

internal extension DeviceUnaryTransformation {
    func functionName<T: FloatingPoint>(forType: T.Type) -> String {
        let baseName = String(describing: self)
        if T.self == Float.self {
            return baseName + "f"
        }
        return baseName
    }
}

public enum DeviceBinaryOperation {
    case addition, subtraction, multiplication, division
}

internal extension DeviceBinaryOperation {
    var macro: String {
        switch self {
        case .addition: return "((_x_) + (_y_))"
        case .subtraction: return "((_x_) - (_y_))"
        case .multiplication: return "((_x_) * (_y_))"
        case .division: return "((_x_) / (_y_))"
        }
    }
}
