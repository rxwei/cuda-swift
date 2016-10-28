//
//  Function.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public enum SharedMemoryBankSize: UInt32 {
    case `default` = 0x00
    case fourBytes = 0x01
    case eightBytes = 0x02
}

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

    init(_ handle: CUfunction) {
        self.handle = handle
    }

    public init(assumingDeviceAddress function: UnsafeRawPointer) {
        handle = OpaquePointer(function)
    }

    public init(assumingDeviceAddress function: UnsafeMutableRawPointer) {
        handle = OpaquePointer(function)
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

        public init(x: Int, y: Int, z: Int, sharedMemory: Int = 0) {
            self.x = x
            self.y = y
            self.z = z
            self.sharedMemory = sharedMemory
        }
    }
    
    public func withUnsafeHandle<Result>(_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

    public struct Arguments {
        var references: [Any] = []
        var addresses: [UnsafeMutableRawPointer?] = []

        public init() {}

        public mutating func append<T>(_ argument: T) {
            references.append(argument)
            addresses.append(&references[references.endIndex-1])
        }

        public mutating func append<T>(_ argument: inout T) {
            addresses.append(&argument)
        }
    }
    
    /// - note:
    /// Needs rewriting
    /// Does not work
    public func launch(withArguments arguments: inout Arguments, inGrid gridSize: GridSize,
                       ofBlocks blockSize: BlockSize, stream: Stream?) throws {
        try ensureSuccess(
            cuLaunchKernel(handle, UInt32(gridSize.x), UInt32(gridSize.y), UInt32(gridSize.z),
                           UInt32(blockSize.x), UInt32(blockSize.y), UInt32(blockSize.z),
                           UInt32(blockSize.sharedMemory), stream?.handle, &arguments.addresses, nil)
        )
        try Context.synchronize()
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
