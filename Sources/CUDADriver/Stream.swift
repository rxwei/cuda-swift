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

    private static func current(withHandle handle: CUstream) -> Stream {
        return Stream.instances[handle]!
    }

    public required init(priority: Int? = nil) throws {
        var handle: CUstream?
        if let priority = priority {
            try ensureSuccess(cuStreamCreateWithPriority(
                &handle, 0, Int32(priority)
            ))
        } else {
            try ensureSuccess(cuStreamCreate(&handle, 0))
        }
        self.handle = handle! // Safe
    }

    deinit {
        Stream.instances.removeValue(forKey: handle)
        cuStreamDestroy_v2(handle)
    }

    open func synchronize() throws {
        try ensureSuccess(cuStreamSynchronize(handle))
    }

    open var priority: Int {
        var priority: Int32 = 0
        cuStreamGetPriority(handle, &priority)
        return Int(priority)
    }

    open func addCallback(_ callback: (Stream?, DriverError?) -> ()) {
        let cuCallback: CUstreamCallback = { handle, result, ptr in
            let callback = unsafeBitCast(ptr, to: ((Stream?, DriverError?) -> ()).self)
            callback(Stream.current(withHandle: handle!),
                     result == CUDA_SUCCESS ? nil : DriverError(result))
        }
        cuStreamAddCallback(handle, cuCallback,
                            unsafeBitCast(callback, to: UnsafeMutableRawPointer.self), 0)
    }

    public func withUnsafeHandle<Result>
        (_ body: (OpaquePointer) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

}
