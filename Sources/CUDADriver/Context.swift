//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCUDA

open class Context {

    let handle: CUcontext

    init(handle: CUcontext) {
        self.handle = handle
    }

    deinit {
        cuCtxDestroy_v2(handle)
    }

    public static var priorityRange: Range<Int> {
        var lowerBound: Int32 = 0
        var upperBound: Int32 = 0
        cuCtxGetStreamPriorityRange(&lowerBound, &upperBound)
        return Int(lowerBound)..<Int(upperBound)
    }

    public init(device: Device) throws {
        var handle: CUcontext?
        try ensureSuccess(cuCtxCreate_v2(&handle, 0, device.unsafeHandle))
        self.handle = handle!
    }

    public func attached() -> AttachedContext {
        return AttachedContext(attaching: self)
    }

    public static func synchronize() {
        cuCtxSynchronize()
    }

}

open class AttachedContext : Context {

    public init(attaching context: Context) {
        var retHandle: CUcontext?
        cuCtxAttach(&retHandle, 0)
        super.init(handle: retHandle!)
    }

    public func detach() {
        cuCtxDetach(handle)
    }
    
}
