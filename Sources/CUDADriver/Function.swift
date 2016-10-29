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

/// Grid of blocks
public struct GridSize {
    public let x: Int, y: Int, z: Int
    
    public init(x: Int, y: Int, z: Int) {
        self.x = x
        self.y = y
        self.z = z
    }
    
    public init(_ count: Int) {
        self.x = count
        self.y = 1
        self.z = 1
    }
}

/// Block of threads
public struct BlockSize {
    public let x: Int, y: Int, z: Int
    /// Shared memory size per thread
    public let memory: Int
    
    public init(x: Int, y: Int, z: Int, sharedMemory: Int = 0) {
        self.x = x
        self.y = y
        self.z = z
        self.memory = sharedMemory
    }
    
    public init(threadCount: Int, memory: Int = 0) {
        self.x = threadCount
        self.y = 1
        self.z = 1
        self.memory = memory
    }
}

public struct ArgumentList {
    private var references: [CUdeviceptr] = []
    private var values: [Any] = []
    var addresses: [UnsafeMutableRawPointer?] = []
    
    public init() {}
    
    public mutating func append<T>(_ address: UnsafeMutableDevicePointer<T>) {
        references.append(address.deviceAddressHandle)
        addresses.append(&references[references.count-1])
    }
    
    public mutating func append<T>(_ argument: inout T) {
        values.append(argument)
        addresses.append(&argument)
    }
}

public enum CachePreference : UInt32 {
    case none = 0x00
    case shared = 0x01
    case l1 = 0x02
    case equal = 0x03
}

public struct Function : CHandleCarrier {
    
    public typealias Handle = CUfunction
    
    let handle: CUfunction
    
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
    
    public init(assumingDeviceAddress address: UnsafeRawPointer) {
        handle = OpaquePointer(address)
    }

    public init(assumingDeviceAddress address: UnsafeMutableRawPointer) {
        handle = OpaquePointer(address)
    }
    
    public func withUnsafeHandle<Result>(_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }
    
    public func launch(with arguments: inout ArgumentList,
                       gridSize: GridSize, blockSize: BlockSize, stream: Stream?) throws {
        try ensureSuccess(
            cuLaunchKernel(handle, UInt32(gridSize.x), UInt32(gridSize.y), UInt32(gridSize.z),
                           UInt32(blockSize.x), UInt32(blockSize.y), UInt32(blockSize.z),
                           UInt32(blockSize.memory), stream?.handle, &arguments.addresses, nil)
        )
        try Context.synchronize()
    }
    
    public func launch(with arguments: inout ArgumentList,
                       blockCount: Int, threadCount: Int,
                       memory: Int, stream: Stream?) throws {
        try launch(with: &arguments,
                   gridSize: GridSize(blockCount),
                   blockSize: BlockSize(threadCount: threadCount, memory: memory),
                   stream: stream)
    }
    
}

precedencegroup CUDAKernelPrecedence {
    associativity: left
    higherThan: TernaryPrecedence
    lowerThan: DefaultPrecedence
}

infix operator <<< : CUDAKernelPrecedence
infix operator >>> : CUDAKernelPrecedence

/// CUDA-like <<<>>> operator
public extension Function {
    
    public static func <<<(lhs: Function, rhs: (Int, Int)) -> (inout ArgumentList) throws -> () {
        return {
            try lhs.launch(with: &$0, blockCount: rhs.0, threadCount: rhs.1, memory: 0, stream: nil)
        }
    }
    
    public static func <<<(lhs: Function, rhs: (Int, Int, Int)) -> (inout ArgumentList) throws -> () {
        return {
            try lhs.launch(with: &$0, blockCount: rhs.0, threadCount: rhs.1, memory: rhs.2, stream: nil)
        }
    }
    
    public static func <<<(lhs: Function, rhs: (Int, Int, Int, Stream)) -> (inout ArgumentList) throws -> () {
        return {
            try lhs.launch(with: &$0, blockCount: rhs.0, threadCount: rhs.1, memory: rhs.2, stream: rhs.3)
        }
    }
    
}

public func >>>(lhs: (inout ArgumentList) throws -> (),
                rhs: (inout ArgumentList)) rethrows {
    try lhs(&rhs)
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
