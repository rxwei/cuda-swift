//
//  Kernel.swift
//  Warp
//
//  Created by Richard Wei on 11/2/16.
//
//

import CUDADriver
@_exported import struct CUDADriver.KernelArgument

/// Kernel operators

public extension KernelArgument {

    public static func array<T>(_ array: inout DeviceArray<T>) -> KernelArgument {
        return array.withUnsafeMutableDevicePointer { ptr in
            self.init(ptr.deviceAddress)
        }
    }

    public static func value<T>(_ value: inout DeviceValue<T>) -> KernelArgument {
        return value.withUnsafeMutableDevicePointer { ptr in
            self.init(ptr.deviceAddress)
        }
    }

}

public protocol KernelDataProtocol {
    static var kernelTypeName: String { get }
}

extension Int : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "long long"
    }
}

extension Float : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "float"
    }
    
}

extension Double : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "double"
    }
}

extension Int8 : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "char"
    }
}
extension Int16 : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "short"
    }
}
extension Int32 : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "long"
    }
}

extension Int64 : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "long long"
    }
}

extension UInt8 : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "unsigned char"
    }
}
extension UInt16 : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "unsigned short"
    }
}
extension UInt32 : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "unsigned long"
    }
}

extension UInt64 : KernelDataProtocol {
    public static var kernelTypeName: String {
        return "unsigned long long"
    }
}
