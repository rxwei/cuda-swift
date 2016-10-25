//
//  Module.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

/// This is a bridge from CUDADriver
@_exported import struct CUDADriver.PTX
@_exported import struct CUDADriver.Function
import enum CUDADriver.Driver
import struct CUDADriver.Device
import class CUDADriver.Context
import class CUDADriver.Module

open class Module : CUDADriver.Module {

    private let context: Context

    public override init(ptx: PTX) throws {
        Driver.initialize()
        /// Create and hold on to the context
        let driverDevice = CUDADriver.Device(atIndex: CUDARuntime.Device.current.index)!
        context = driverDevice.makeContext()
        try super.init(ptx: ptx)
    }

    deinit {
        Context.synchronize()
    }
    
}
