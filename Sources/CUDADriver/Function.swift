//
//  Function.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public struct Function : CHandleCarrier {

    public typealias Handle = CUfunction

    let handle: CUfunction

    public enum CachePreference : UInt32 {
        case none = 0x00
        case shared = 0x01
        case l1 = 0x02
        case equal = 0x03
    }

    public var cachePreference: CachePreference = .none {
        didSet {
            cuFuncSetCacheConfig(
                handle, CUfunc_cache(rawValue: cachePreference.rawValue)
            )
        }
    }

    public var sharedMemoryBankSize: SharedMemoryBankSize = .default {
        didSet {
            cuFuncSetSharedMemConfig(
                handle,
                CUsharedconfig(rawValue: sharedMemoryBankSize.rawValue)
            )
        }
    }

    public init(_ handle: CUfunction) {
        self.handle = handle
    }

    /// Grid of blocks
    public struct GridSize {
        public let x: Int, y: Int, z: Int

        public init(x: Int, y: Int, z: Int) {
            self.x = x
            self.y = y
            self.z = z
        }
    }

    /// Block of threads
    public struct BlockSize {
        public let x: Int, y: Int, z: Int
        /// Shared memory size per thread
        public let sharedMemory: Int
        
        public init(x: Int, y: Int, z: Int, sharedMemorySize: Int) {
            self.x = x
            self.y = y
            self.z = z
            self.sharedMemory = sharedMemorySize
        }
    }
    
    public func withUnsafeHandle<Result>(_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

    /// - note:
    /// Needs rewriting
    /// Does not work
    public func launch(withArguments arguments: [Any], inGrid gridSize: GridSize,
                       ofBlocks blockSize: BlockSize, stream: Stream?) throws {
        try arguments.withUnsafeBufferPointer { ptr in
            let argPtr = unsafeBitCast(ptr.baseAddress, to: UnsafeMutablePointer<UnsafeMutableRawPointer?>.self)
            try ensureSuccess(
                cuLaunchKernel(handle, UInt32(gridSize.x), UInt32(gridSize.y), UInt32(gridSize.z),
                               UInt32(blockSize.x), UInt32(blockSize.y), UInt32(blockSize.z),
                               UInt32(blockSize.sharedMemory), stream?.handle ?? nil, argPtr, nil)
            )
        }
    }

}

public extension Function {

    public var maxThreadsPerBlock: Int {
        var maxThreads: Int32 = 0
        cuFuncGetAttribute(&maxThreads,
                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                           handle)
        return Int(maxThreads)
    }

    public var sharedSize: Int {
        var maxThreads: Int32 = 0
        cuFuncGetAttribute(&maxThreads,
                           CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                           handle)
        return Int(maxThreads)
    }

    public var userConstSize: Int {
        var maxThreads: Int32 = 0
        cuFuncGetAttribute(&maxThreads,
                           CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                           handle)
        return Int(maxThreads)
    }
    
    public var localSize: Int {
        var maxThreads: Int32 = 0
        cuFuncGetAttribute(&maxThreads,
                           CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                           handle)
        return Int(maxThreads)
    }

    public var registerCount: Int {
        var maxThreads: Int32 = 0
        cuFuncGetAttribute(&maxThreads,
                           CU_FUNC_ATTRIBUTE_NUM_REGS,
                           handle)
        return Int(maxThreads)
    }
    
    public var PTXVersion: Int {
        var maxThreads: Int32 = 0
        cuFuncGetAttribute(&maxThreads,
                           CU_FUNC_ATTRIBUTE_PTX_VERSION,
                           handle)
        return Int(maxThreads)
    }
    
    public var binaryVersion: Int {
        var maxThreads: Int32 = 0
        cuFuncGetAttribute(&maxThreads,
                           CU_FUNC_ATTRIBUTE_BINARY_VERSION,
                           handle)
        return Int(maxThreads)
    }

}
