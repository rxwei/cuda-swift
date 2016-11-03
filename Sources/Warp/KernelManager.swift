//
//  KernelManager.swift
//  CUDA
//
//  Created by Richard Wei on 11/3/16.
//
//

import CUDARuntime
import NVRTC
@_exported import struct CUDARuntime.Device

public class KernelManager {

    fileprivate static var instances: [Int : KernelManager] = [:]

    open class func main(on device: Device) -> KernelManager {
        if let manager = instances[device.index] {
            return manager
        }
        let manager = try! KernelManager(device: device)
        instances[device.index] = manager
        return manager
    }

    open static var compileOptions: [CompileOption] = [
        .useFastMath
    ]

    let device: Device

    fileprivate var modules: [Module] = []

    public init(device: Device) throws {
        self.device = device
        try loadKernels()
    }

    private func loadKernels() throws {
        let intrinsicOptions: [CompileOption] = [
            .computeCapability(device.computeCapability)
        ]
        let module = try Module(
            source: builtinKernelSource,
            compileOptions: intrinsicOptions + KernelManager.compileOptions
        )
        modules.append(module)
    }

}
