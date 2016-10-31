//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCUDA

public struct Context : CHandleCarrier {

    public typealias Handle = CUcontext?

    var handle: CUcontext?

    public struct Options : OptionSet {
        public static let autoScheduling = 0x00
        public static let spinScheduling = 0x01
        public static let blockingSyncScheduling = 0x02
        
        @available(*, deprecated, message: "Deprecated as of CUDA 4.0. Use .blockingSyncScheduling instead.")
        public static let blockingSync = 0x04
        
        /// Mask for scheduling bits, not for external use
        static let schedulingMask = 0x07

        public static let mappedPinnedAllocations = 0x08
        public static let keepLocalMemory = 0x10

        /// Mask for all flag bits, not for external use
        private static let flagsMask = 0x1f

        public let rawValue: UInt32

        public init(rawValue: UInt32) {
            self.rawValue = rawValue
        }
    }

    public static func begin(on device: Device, options: Options? = nil) -> Context {
        var ctxHandle: CUcontext?
        !!cuCtxCreate_v2(&ctxHandle, 0, device.handle)
        if let options = options {
            !!cuDevicePrimaryCtxSetFlags(device.handle, options.rawValue)
        }
        return Context(ctxHandle!)
    }

    public mutating func end() {
        !!cuCtxDestroy_v2(handle)
        handle = nil
    }

    public static var currentDevice: Device {
        var deviceHandle: CUdevice = 0
        !!cuCtxGetDevice(&deviceHandle)
        return Device(deviceHandle)
    }

    public static var priorityRange: Range<Int> {
        var lowerBound: Int32 = 0
        var upperBound: Int32 = 0
        !!cuCtxGetStreamPriorityRange(&lowerBound, &upperBound)
        return Int(lowerBound)..<Int(upperBound)
    }

    init(_ handle: CUcontext) {
        self.handle = handle
    }

    /// Binds the specified CUDA context to the calling CPU thread.
    /// If there exists a CUDA context stack on the calling CPU thread,
    /// this will replace the top of that stack with self.
    public static var current: Context? {
        set {
            !!cuCtxSetCurrent(newValue?.handle)
        }
        get {
            var handle: CUcontext?
            !!cuCtxGetCurrent(&handle)
            return handle.flatMap { Context($0) }
        }
    }

    /// Pushes self onto the CPU thread's stack of current contexts.
    /// The specified context becomes the CPU thread's current context,
    /// so all CUDA functions that operate on the current context are 
    /// affected.
    public func pushToThread() {
        !!cuCtxPushCurrent_v2(handle)
    }

    /// Pops the current CUDA context from the CPU thread and returns it
    /// - returns: the popped context, if any
    public static func popFromThread() -> Context? {
        var handle: CUcontext?
        !!cuCtxPopCurrent_v2(&handle)
        return handle.flatMap { Context($0) }
    }

    public static func synchronize() throws {
        try ensureSuccess(cuCtxSynchronize())
    }
    
    public func withUnsafeHandle<Result>
        (_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

}
