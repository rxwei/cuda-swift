//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCUDA

open class Context {

    public let global = Context(device: Device.default)

    let handle: CUcontext

    init(handle: CUcontext) {
        self.handle = handle
    }

    deinit {
        cuCtxDestroy_v2(handle)
    }

    public init(device: Device) {
        var handle: CUcontext?
        cuCtxCreate_v2(&handle, 0, device.handle)
        self.handle = handle!
    }

    public func attached() -> AttachedContext {
        return AttachedContext(attaching: self)
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
