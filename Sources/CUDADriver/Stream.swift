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
    let owning: Bool

    private static var instances: [CUstream : Stream] = [:]
    
    class func current(with handle: CUstream) -> Stream {
        return Stream.instances[handle]!
    }

    public init() {
        owning = true
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
        owning = true
        self.handle = handle! // Safe
        Stream.instances[self.handle] = self
    }

    /// Unsafely reference a stream handle.
    /// This is *not* intended for public use. Currently it acts as a bridge
    /// between Runtime API and Driver API, specifically for `Function.launch`
    /// to take a runtime stream argument.
    /// - note: This will be removed once runtmie has its own `Function` logic
    /// implemented.
    public init!(unsafelyReferencing handle: Handle) {
        owning = false
        self.handle = handle
        Stream.instances[self.handle] = self
    }

    deinit {
        if owning {
            Stream.instances.removeValue(forKey: handle)
            cuStreamDestroy_v2(handle)
        }
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
            callback?(Stream.current(with: handle!),
                      result == CUDA_SUCCESS ? nil : DriverError(result))
        }
        callbacks.append(callback)
        cuStreamAddCallback(handle, cuCallback,
                            &callbacks[callbacks.endIndex-1], 0)
    }

    public func withUnsafeHandle<Result>
        (_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

}
