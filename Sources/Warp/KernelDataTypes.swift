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

    public static func array<T>(_ array: inout DeviceArray<T>) -> KernelArgument {
        return array.withUnsafeMutableDevicePointer { ptr in
            self.init(ptr.deviceAddress)
        }
    }

    public static func constantArray<T>(_ array: DeviceArray<T>) -> KernelArgument {
        return array.withUnsafeDevicePointer { ptr in
            self.init(ptr.deviceAddress)
        }
    }

    public static func valuePointer<T>(_ value: inout DeviceValue<T>) -> KernelArgument {
        return value.withUnsafeMutableDevicePointer { ptr in
            self.init(ptr.deviceAddress)
        }
    }

    public static func constantValuePointer<T>(_ value: DeviceValue<T>) -> KernelArgument {
        return value.withUnsafeDevicePointer { ptr in
            self.init(ptr.deviceAddress)
        }
    }

    public static func value<T: KernelDataProtocol>(_ value: T) -> KernelArgument {
        return self.init(value)
    }

}

public enum KernelDataType : String {
    case float            = "float"
    case double           = "double"
    case char             = "char"
    case short            = "short"
    case int              = "int"
    case longLong         = "long long"
    case unsignedChar     = "unsigned char"
    case unsignedShort    = "unsigned short"
    case unsignedInt      = "unsigned int"
    case unsignedLongLong = "unsigned long long"
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
