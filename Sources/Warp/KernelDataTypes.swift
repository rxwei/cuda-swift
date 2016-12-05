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

    public static func pointer<T: MutableDeviceAddressable>(to value: inout T) -> KernelArgument {
        return self.init(value.unsafeMutableDevicePointer.deviceAddress)
    }

    public static func constPointer<T: DeviceAddressable>(to value: T) -> KernelArgument {
        return self.init(value.unsafeDevicePointer.deviceAddress)
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
    public static let kernelDataType: KernelDataType = .longLong
}

extension UInt : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .unsignedLongLong
}

extension Float : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .float
}

extension Double : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .double
}

extension Int8 : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .char
}

extension Int16 : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .short
}

extension Int32 : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .int
}

extension Int64 : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .longLong
}

extension UInt8 : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .unsignedChar
}

extension UInt16 : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .unsignedShort
}

extension UInt32 : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .unsignedInt
}

extension UInt64 : KernelDataProtocol {
    public static let kernelDataType: KernelDataType = .unsignedLongLong
}
