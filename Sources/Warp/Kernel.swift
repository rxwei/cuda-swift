//
//  Kernel.swift
//  Warp
//
//  Created by Richard Wei on 11/2/16.
//
//

import class NVRTC.Program
import class NVRTC.Compiler
import struct CUDARuntime.Device
import CUDADriver

/// Kernel operators

public extension KernelArgument {

    public static func array<T>(_ array: inout DeviceArray<T>) -> KernelArgument {
        return array.withUnsafeMutableDevicePointer { ptr in
            self.init(unsafeAddress: ptr.deviceAddress)
        }
    }

    public static func reference<T>(_ value: inout DeviceValue<T>) -> KernelArgument {
        return value.withUnsafeMutableDevicePointer { ptr in
            self.init(unsafeAddress: ptr.deviceAddress)
        }
    }

}
