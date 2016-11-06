//
//  Function.swift
//  CUDA
//
//  Created by Richard Wei on 11/1/16.
//
//

import CCUDARuntime
import struct CUDADriver.Function
import class CUDADriver.Stream
import enum CUDADriver.SharedMemoryBankSize
import struct CUDADriver.GridSize
import struct CUDADriver.BlockSize
import enum CUDADriver.CachePreference
import struct CUDADriver.KernelArgument


public struct Function {

    fileprivate let address: UnsafeRawPointer

    fileprivate let attributes: cudaFuncAttributes

    public var cachePreference: CachePreference = .none {
        didSet {
            cudaFuncSetCacheConfig(address, cudaFuncCache(cachePreference.rawValue))
        }
    }

    public var sharedMemoryBankSize: SharedMemoryBankSize = .default {
        didSet {
            cudaFuncSetSharedMemConfig(address, cudaSharedMemConfig(sharedMemoryBankSize.rawValue))
        }
    }

    init(unsafeAddress address: UnsafeRawPointer) {
        self.address = address
        var attributes = cudaFuncAttributes()
        cudaFuncGetAttributes(&attributes, address)
        self.attributes = attributes
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
