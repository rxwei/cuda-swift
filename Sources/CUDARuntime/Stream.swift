//
//  Stream.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

import CCUDARuntime
import protocol CUDADriver.CHandleCarrier

open class Stream : CHandleCarrier {

    public typealias Handle = cudaStream_t

    private static var instances: [Handle : Stream] = [:]

    class func current(with handle: Handle) -> Stream? {
        return instances[handle]
    }

    let handle: cudaStream_t

    public init() {
        var handle: Handle?
        !!cudaStreamCreate(&handle)
        self.handle = handle!
        Stream.instances[self.handle] = self
    }

    public init?(priority: Int) {
        var handle: Handle?
        guard
            let _ = try? ensureSuccess(cudaStreamCreateWithPriority(&handle, 0, Int32(priority)))
            else { return nil }
        self.handle = handle! // Safe
        Stream.instances[self.handle] = self
    }

    deinit {
        Stream.instances.removeValue(forKey: handle)
        cudaStreamDestroy(handle)
    }

    open func synchronize() {
        !!cudaStreamSynchronize(handle)
    }

    open var priority: Int {
        var priority: Int32 = 0
        !!cudaStreamGetPriority(handle, &priority)
        return Int(priority)
    }

    /// Buffer that keeps track of all callbacks. This is needed because we need
    /// to follow the C-convention, i.e. no local capture, in CUDA's callback
    /// function.
    private var callbacks: [(Stream?, RuntimeError?) -> ()] = []

    open func addCallback(_ callback: @escaping (Stream?, RuntimeError?) -> ()) {
        let cuCallback: cudaStreamCallback_t = { handle, result, userDataPtr in
            let callback = userDataPtr?.assumingMemoryBound(
                to: ((Stream?, RuntimeError?) -> ()).self).pointee
            callback?(handle.flatMap(Stream.current(with:)),
                      result == cudaSuccess ? nil : RuntimeError(result))
        }
        callbacks.append(callback)
        !!cudaStreamAddCallback(handle, cuCallback, &callbacks[callbacks.endIndex-1], 0)
    }

    public func withUnsafeHandle<Result>
        (_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }
    
}
