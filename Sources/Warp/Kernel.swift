//
//  Kernel.swift
//  Warp
//
//  Created by Richard Wei on 11/2/16.
//
//

@_exported import class CUDADriver.Module
@_exported import struct CUDARuntime.Function
@_exported import class CUDARuntime.Stream
@_exported import struct CUDARuntime.Device
@_exported import enum CUDADriver.SharedMemoryBankSize
@_exported import struct CUDADriver.GridSize
@_exported import struct CUDADriver.BlockSize
@_exported import enum CUDADriver.CachePreference
@_exported import struct CUDADriver.KernelArgument


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

public func <<<(lhs: Function, rhs: (Int, Int, Int, Stream)) -> ([KernelArgument]) throws -> () {
    return { (args: [KernelArgument]) in
        try lhs.launch(with: args, blockCount: rhs.0, threadCount: rhs.1, memory: rhs.2, stream: rhs.3)
    }
}

public func >>>(lhs: ([KernelArgument]) throws -> (), rhs: [KernelArgument]) rethrows {
    try lhs(rhs)
}
