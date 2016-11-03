//
//  Kernel.swift
//  Warp
//
//  Created by Richard Wei on 11/2/16.
//
//

@_exported import class NVRTC.Program
@_exported import class NVRTC.Compiler
@_exported import struct CUDARuntime.Device
@_exported import class CUDARuntime.Module
@_exported import struct CUDARuntime.Kernel
import struct CUDADriver.Function
@_exported import class CUDARuntime.Stream
@_exported import enum CUDARuntime.SharedMemoryBankSize
@_exported import struct CUDARuntime.GridSize
@_exported import struct CUDARuntime.BlockSize
@_exported import enum CUDARuntime.CachePreference
@_exported import struct CUDARuntime.KernelArgument

/// Kernel operators

precedencegroup CUDAKernelPrecedence {
    associativity: left
    higherThan: TernaryPrecedence
    lowerThan: DefaultPrecedence
}
infix operator <<< : CUDAKernelPrecedence
infix operator >>> : CUDAKernelPrecedence

public func <<<(lhs: Function, rhs: (Int, Int)) -> ([KernelArgument]) throws -> () {
    return { (args: [KernelArgument]) in
        try lhs.launch(with: args, blockCount: rhs.0, threadCount: rhs.1, memory: 0, stream: nil)
    }
}

public func <<<(lhs: Function, rhs: (Int, Int, Int)) -> ([KernelArgument]) throws -> () {
    return { (args: [KernelArgument]) in
        try lhs.launch(with: args, blockCount: rhs.0, threadCount: rhs.1, memory: rhs.2, stream: nil)
    }
}

public func <<<(lhs: Kernel, rhs: (Int, Int)) -> ([KernelArgument]) throws -> () {
    return { (args: [KernelArgument]) in
        try lhs.launch(with: args, blockCount: rhs.0, threadCount: rhs.1, memory: 0, stream: nil)
    }
}

public func <<<(lhs: Kernel, rhs: (Int, Int, Int)) -> ([KernelArgument]) throws -> () {
    return { (args: [KernelArgument]) in
        try lhs.launch(with: args, blockCount: rhs.0, threadCount: rhs.1, memory: rhs.2, stream: nil)
    }
}

public func <<<(lhs: Kernel, rhs: (Int, Int, Int, Stream)) -> ([KernelArgument]) throws -> () {
    return { (args: [KernelArgument]) in
        try lhs.launch(with: args, blockCount: rhs.0, threadCount: rhs.1, memory: rhs.2, stream: rhs.3)
    }
}

public func >>>(lhs: ([KernelArgument]) throws -> (), rhs: [KernelArgument]) rethrows {
    try lhs(rhs)
}

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
