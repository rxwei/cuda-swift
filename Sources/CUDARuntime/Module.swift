//
//  Module.swift
//  CUDA
//
//  Created by Richard Wei on 11/3/16.
//
//

import CUDADriver

open class Module : CUDADriver.Module {

    private var context: Context

    public override init(ptx: PTX) throws {
        Driver.initialize()
        /// Create and hold on to the context
        guard let driverDevice = CUDARuntime.Device.current.flatMap ({ device in
            CUDADriver.Device(atIndex: device.index)
        }) else {
            throw RuntimeError.noDevice
        }
        context = Context.begin(on: driverDevice)
        try super.init(ptx: ptx)
    }

    open func kernel(named name: String) -> Kernel? {
        let driverFunc = super.function(named: name)
        return driverFunc?.withUnsafeHandle { handle in
            Kernel(unsafeAddress: UnsafeRawPointer(handle))
        }
    }

    deinit {
        context.end()
    }
    
}
