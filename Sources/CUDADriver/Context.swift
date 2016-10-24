//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCUDA

open class Context : CHandleCarrier {

    public typealias Handle = CUcontext

    let handle: CUcontext

    private static var instances: [CUcontext : Context] = [:]

    deinit {
        Context.instances.removeValue(forKey: handle)
        cuCtxDestroy_v2(handle)
    }

    open class var priorityRange: Range<Int> {
        var lowerBound: Int32 = 0
        var upperBound: Int32 = 0
        cuCtxGetStreamPriorityRange(&lowerBound, &upperBound)
        return Int(lowerBound)..<Int(upperBound)
    }

    public init(device: Device) throws {
        var handle: CUcontext?
        try ensureSuccess(cuCtxCreate_v2(&handle, 0, device.handle))
        self.handle = handle!
    }

    /// Binds the specified CUDA context to the calling CPU thread.
    /// If there exists a CUDA context stack on the calling CPU thread,
    /// this will replace the top of that stack with self.
    open func bindToCallingCPUThread() throws {
        try ensureSuccess(cuCtxPushCurrent_v2(handle))
    }

    /// Pushes the given context ctx onto the CPU thread's stack of current
    /// contexts. The specified context becomes the CPU thread's current
    /// context, so all CUDA functions that operate on the current context 
    /// are affected.
    open func push() throws {
        cuCtxPushCurrent_v2(handle)
    }

    /// Pops the current CUDA context from the CPU thread and returns it
    /// - returns: the popped context, if any
    open class func pop() throws -> Context? {
        var handle: CUcontext?
        try ensureSuccess(cuCtxPopCurrent_v2(&handle))
        return handle == nil ? nil : instances[handle!]!
    }

    open class func synchronize() {
        cuCtxSynchronize()
    }
    
    public func withUnsafeHandle<Result>
        (_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

}
