//
//  Stream.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public final class Stream {

    public var handle: CUstream

    private static var streams: [CUstream : Stream] = [:]

    private static func stream(withHandle handle: CUstream) -> Stream {
        return Stream.streams[handle]!
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
        Stream.streams.removeValue(forKey: handle)
        cuStreamDestroy_v2(handle)
    }

    public func synchronize() throws {
        try ensureSuccess(cuStreamSynchronize(handle))
    }

    public var priority: Int {
        var priority: Int32 = 0
        cuStreamGetPriority(handle, &priority)
        return Int(priority)
    }

    public func addCallback(_ callback: @escaping (Stream?, DriverError?) -> ()) {
        let cuCallback: CUstreamCallback = { handle, result, ptr in
            let callback = unsafeBitCast(ptr, to: ((Stream?, DriverError?) -> ()).self)
            callback(Stream.stream(withHandle: handle!),
                     result == CUDA_SUCCESS ? nil : DriverError(result))
        }
        cuStreamAddCallback(handle, cuCallback,
                            unsafeBitCast(callback, to: UnsafeMutableRawPointer.self), 0)
    }

}
