//
//  Stream.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

/// - Richard's thoughts:
/// Is the following really needed? Perhaps we can directly use the Stream from driver.
/// |- No. Driver requires `cuInit(0)`, i.e. `Driver.initialize()`. What a shame!

import CCUDARuntime
import protocol CUDADriver.CHandleCarrier

open class Stream : CHandleCarrier {

    public typealias Handle = cudaStream_t

    private static var instances: [Handle : Stream] = [:]

    open class func current(with handle: Handle) -> Stream? {
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
        do {
            try ensureSuccess(cudaStreamCreateWithPriority(&handle, 0, Int32(priority)))
        } catch {
            return nil
        }
        self.handle = handle! // Safe
        Stream.instances[self.handle] = self
    }

    deinit {
        Stream.instances.removeValue(forKey: handle)
        !!cudaStreamDestroy(handle)
    }

    open func synchronize() throws {
        !!cudaStreamSynchronize(handle)
    }

    open var priority: Int {
        var priority: Int32 = 0
        !!cudaStreamGetPriority(handle, &priority)
        return Int(priority)
    }

    open func addCallback(_ callback: (Stream?, RuntimeError?) -> ()) {
        let cuCallback: cudaStreamCallback_t = { handle, result, ptr in
            let callback = unsafeBitCast(ptr, to: ((Stream?, RuntimeError?) -> ()).self)
            callback(Stream.current(with: handle!),
                     result == cudaSuccess ? nil : RuntimeError(result))
        }
        cudaStreamAddCallback(handle, cuCallback,
                              unsafeBitCast(callback, to: UnsafeMutableRawPointer.self), 0)
    }

    public func withUnsafeHandle<Result>(_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }
    
}
