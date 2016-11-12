//
//  Stream.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

open class Stream : CHandleCarrier {

    public typealias Handle = CUstream

    let handle: CUstream

    private static var instances: [CUstream : Stream] = [:]
    
    class func current(with handle: CUstream) -> Stream {
        return Stream.instances[handle]!
    }

    public init() {
        var handle: CUstream?
        !!cuStreamCreate(&handle, 0)
        self.handle = handle! // Safe
        Stream.instances[self.handle] = self
    }

    public init?(priority: Int) {
        var handle: CUstream?
        do {
            try ensureSuccess(cuStreamCreateWithPriority(&handle, 0, Int32(priority)))
        } catch {
            return nil
        }
        self.handle = handle! // Safe
        Stream.instances[self.handle] = self
    }

    deinit {
        Stream.instances.removeValue(forKey: handle)
        cuStreamDestroy_v2(handle)
    }

    open func synchronize() {
        !!cuStreamSynchronize(handle)
    }

    open var priority: Int {
        var priority: Int32 = 0
        cuStreamGetPriority(handle, &priority)
        return Int(priority)
    }

    /// Buffer that keeps track of all callbacks. This is needed because we need
    /// to follow the C-convention, i.e. no local capture, in CUDA's callback
    /// function.
    private var callbacks: [(Stream?, DriverError?) -> ()] = []

    open func addCallback(_ callback: @escaping (Stream?, DriverError?) -> ()) {
        let cuCallback: CUstreamCallback = { handle, result, userDataPtr in
            let callback = userDataPtr?.assumingMemoryBound(
                to: ((Stream?, DriverError?) -> ()).self).pointee
            callback?(handle.flatMap(Stream.current(with:)),
                      result == CUDA_SUCCESS ? nil : DriverError(result))
        }
        callbacks.append(callback)
        cuStreamAddCallback(handle, cuCallback, &callbacks[callbacks.endIndex-1], 0)
    }

    public func withUnsafeHandle<Result>
        (_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

}
