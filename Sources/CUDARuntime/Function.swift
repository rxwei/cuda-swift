//
//  Function.swift
//  CUDA
//
//  Created by Richard Wei on 11/1/16.
//
//

import CCUDARuntime
import struct CUDADriver.Function
@_exported import enum CUDADriver.SharedMemoryBankSize
@_exported import struct CUDADriver.GridSize
@_exported import struct CUDADriver.BlockSize
@_exported import enum CUDADriver.CachePreference
@_exported import struct CUDADriver.KernelArgument

public struct Function {

    let address: UnsafeRawPointer

    fileprivate let attributes: cudaFuncAttributes

    public init(_ driverFunction: CUDADriver.Function) {
        address = UnsafeRawPointer(driverFunction.withUnsafeHandle{$0})
        var attr = cudaFuncAttributes()
        !!cudaFuncGetAttributes(&attr, address)
        attributes = attr
    }

    public var cachePreference: CachePreference = .none {
        didSet {
            !!cudaFuncSetCacheConfig(address, cudaFuncCache(cachePreference.rawValue))
        }
    }

    public var sharedMemoryBankSize: SharedMemoryBankSize = .default {
        didSet {
            !!cudaFuncSetSharedMemConfig(address, cudaSharedMemConfig(sharedMemoryBankSize.rawValue))
        }
    }

    func withUnsafeDeviceAddress<Result>
        (_ body: (UnsafeRawPointer) throws -> Result) rethrows -> Result {
        return try body(address)
    }

    public func launch(with arguments: [KernelArgument],
                       gridSize: GridSize, blockSize: BlockSize, stream: Stream?) throws {
        var arguments = arguments
        var addresses = (0..<arguments.count).map{i in arguments[i].unsafeAddress}
        try ensureSuccess(
            cudaLaunchKernel(address,
                             dim3(x: UInt32(blockSize.x),
                                  y: UInt32(blockSize.y),
                                  z: UInt32(blockSize.z)),
                             dim3(x: UInt32(blockSize.x),
                                  y: UInt32(blockSize.y),
                                  z: UInt32(blockSize.z)),
                             &addresses,
                             blockSize.memory, stream?.handle)
        )
    }

    public func launch(with arguments: [KernelArgument],
                       blockCount: Int, threadCount: Int,
                       memory: Int, stream: Stream?) throws {
        try launch(with: arguments,
                   gridSize: GridSize(blockCount: blockCount),
                   blockSize: BlockSize(threadCount: threadCount, memory: memory),
                   stream: stream)
    }

}

public extension Function {
    
    public var maxThreadsPerBlock: Int {
        return Int(attributes.maxThreadsPerBlock)
    }

    public var sharedSize: Int {
        return Int(attributes.sharedSizeBytes)
    }
    
    public var constSize: Int {
        return Int(attributes.constSizeBytes)
    }
    
    public var localSize: Int {
        return Int(attributes.localSizeBytes)
    }

    public var registerCount: Int {
        return Int(attributes.numRegs)
    }
    
    public var ptxVersion: Int {
        return Int(attributes.ptxVersion)
    }

    public var binaryVersion: Int {
        return Int(attributes.binaryVersion)
    }

    public var isCacheModeCAEnabled: Bool {
        return attributes.cacheModeCA != 0
    }

}

infix operator <<< : CUDAKernelPrecedence
infix operator >>> : CUDAKernelPrecedence

@inline(__always)
public func <<<(lhs: CUDARuntime.Function, rhs: (CUDARuntime.Function) throws -> ()) rethrows {
    try rhs(lhs)
}

@inline(__always)
public func >>>(lhs: (Int, Int), rhs: [KernelArgument]) -> (CUDARuntime.Function) throws -> () {
    return { f in
        try f.launch(with: rhs, blockCount: lhs.0, threadCount: lhs.1, memory: 0, stream: nil)
    }
}

@inline(__always)
public func >>>(lhs: (Int, Int, Int, Stream?), rhs: [KernelArgument]) -> (CUDARuntime.Function) throws -> () {
    return { f in
        try f.launch(with: rhs, blockCount: lhs.0, threadCount: lhs.1, memory: lhs.2, stream: lhs.3)
    }
}

public extension KernelArgument {

    public static func pointer<T>(_ pointer: UnsafeMutableDevicePointer<T>) -> KernelArgument {
        return self.init(pointer.deviceAddress)
    }

    public static func pointer<T>(_ pointer: UnsafeDevicePointer<T>) -> KernelArgument {
        return self.init(pointer.deviceAddress)
    }

}
