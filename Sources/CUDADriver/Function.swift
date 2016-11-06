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
    
    fileprivate let handle: CUfunction
    
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
    
    public func withUnsafeHandle<Result>(_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }
    
    public func launch(with arguments: [KernelArgument],
                       gridSize: GridSize, blockSize: BlockSize, stream: Stream?) throws {
        var arguments = arguments
        var addresses = (0..<arguments.count).map{i in arguments[i].unsafeAddress}
        try ensureSuccess(
            cuLaunchKernel(handle, UInt32(gridSize.x), UInt32(gridSize.y), UInt32(gridSize.z),
                           UInt32(blockSize.x), UInt32(blockSize.y), UInt32(blockSize.z),
                           UInt32(blockSize.memory), stream?.handle, &addresses, nil)
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
    
    public var constSize: Int {
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
    
    public var ptxVersion: Int {
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

/// A variation of CUDA <<<>>> Operators
precedencegroup CUDAKernelPrecedence {
    associativity: right
    higherThan: TernaryPrecedence
    lowerThan: DefaultPrecedence
}
infix operator <<< : CUDAKernelPrecedence
infix operator >>> : CUDAKernelPrecedence

@inline(__always)
public func <<<(lhs: Function, rhs: (Function) throws -> ()) rethrows {
    try rhs(lhs)
}

@inline(__always)
public func >>>(lhs: (Int, Int), rhs: [KernelArgument]) -> (Function) throws -> () {
    return { (f: Function) throws -> () in
        try f.launch(with: rhs, blockCount: lhs.0, threadCount: lhs.1, memory: 0, stream: nil)
    }
}

@inline(__always)
public func >>>(lhs: (Int, Int, Int, Stream?), rhs: [KernelArgument]) -> (Function) throws -> () {
    return { (f: Function) throws -> () in
        try f.launch(with: rhs, blockCount: lhs.0, threadCount: lhs.1, memory: lhs.2, stream: lhs.3)
    }
}

public enum SharedMemoryBankSize: UInt32 {
    case `default` = 0x00
    case fourBytes = 0x01
    case eightBytes = 0x02
}

public enum CachePreference : UInt32 {
    case none = 0x00
    case shared = 0x01
    case l1 = 0x02
    case equal = 0x03
}

/// Grid of blocks
public struct GridSize {
    /// Dimensions
    public let x: Int, y: Int, z: Int
    
    public init(x: Int, y: Int, z: Int) {
        self.x = x
        self.y = y
        self.z = z
    }

    public init(blockCount: Int) {
        self.x = blockCount
        self.y = 1
        self.z = 1
    }
}

/// Block of threads
public struct BlockSize {
    /// Dimensions
    public let x: Int, y: Int, z: Int
    /// Shared memory size per thread
    public let memory: Int
    
    public init(x: Int, y: Int, z: Int, memory: Int = 0) {
        self.x = x
        self.y = y
        self.z = z
        self.memory = memory
    }
    
    public init(threadCount: Int, memory: Int = 0) {
        self.x = threadCount
        self.y = 1
        self.z = 1
        self.memory = memory
    }
}

public struct KernelArgument {
    
    private var value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public var unsafeAddress: UnsafeMutableRawPointer? {
        mutating get {
            return UnsafeMutableRawPointer(&self.value)
        }
    }

    public static func char(_ number: Int8) -> KernelArgument {
        return self.init(number)
    }

    public static func short(_ number: Int16) -> KernelArgument {
        return self.init(number)
    }

    public static func int(_ number: Int32) -> KernelArgument {
        return self.init(number)
    }

    public static func long(_ number: Int64) -> KernelArgument {
        return self.init(number)
    }

    public static func longLong(_ number: Int64) -> KernelArgument {
        return self.init(number)
    }

    public static func unsignedChar(_ number: UInt8) -> KernelArgument {
        return self.init(number)
    }

    public static func unsignedShort(_ number: UInt16) -> KernelArgument {
        return self.init(number)
    }

    public static func unsignedInt(_ number: UInt32) -> KernelArgument {
        return self.init(number)
    }

    public static func unsignedLong(_ number: UInt32) -> KernelArgument {
        return self.init(number)
    }

    public static func unsignedLongLong(_ number: UInt64) -> KernelArgument {
        return self.init(number)
    }

    public static func float(_ number: Float) -> KernelArgument {
        return self.init(number)
    }

    public static func double(_ number: Double) -> KernelArgument {
        return self.init(number)
    }

    public static func pointer<T>(_ address: UnsafeMutableDevicePointer<T>) -> KernelArgument {
        return self.init(address.deviceAddress)
    }

}
