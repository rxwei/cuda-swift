//
//  KernelManager.swift
//  CUDA
//
//  Created by Richard Wei on 11/3/16.
//
//

import CUDARuntime
import class CUDADriver.Module
import struct CUDADriver.Function
import NVRTC

final class KernelManager {

    fileprivate static var instances: [Int : KernelManager] = [:]

    static func main(on device: Device) -> KernelManager {
        if let manager = instances[device.index] {
            return manager
        }
        let manager = try! KernelManager(device: device)
        instances[device.index] = manager
        return manager
    }

    static var compileOptions: [CompileOption] = [
        .useFastMath
    ]

    let device: Device

    fileprivate var modules: [Module] = []

    public init(device: Device) throws {
        self.device = device
    }

}
