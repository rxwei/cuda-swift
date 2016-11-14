//
//  KernelManager.swift
//  CUDA
//
//  Created by Richard Wei on 11/3/16.
//
//

import struct CUDARuntime.Device
import CUDADriver
import NVRTC

final class KernelManager {

    private static var instances: [KernelManager?] = Array(repeating: nil, count: Device.count)

    static func shared(on device: Device) -> KernelManager {
        if let manager = instances[device.index] {
            return manager
        }
        let manager = KernelManager(device: device)
        instances[device.index] = manager
        return manager
    }

    let device: Device

    struct ModuleCacheKey : Equatable, Hashable {
        let type: KernelDataType
        let source: KernelSource

        static func ==(lhs: ModuleCacheKey, rhs: ModuleCacheKey) -> Bool {
            return lhs.type == rhs.type && lhs.source == rhs.source
        }

        var hashValue: Int {
            return type.hashValue ^ source.hashValue
        }
    }

    fileprivate var modules: [ModuleCacheKey : (Module, Function)] = Dictionary(minimumCapacity: 32)

    init(device: Device) {
        self.device = device
    }
    
    func kernel<T: KernelDataProtocol>(_ source: KernelSource, forType: T.Type) -> Function {
        /// Get cached function
        let key = ModuleCacheKey(type: T.kernelDataType, source: source)
        if let (_, function) = modules[key] {
            return function
        }
        /// If not cached, compile using NVRTC
        let ptx = try! Compiler.compile(
            source.rawValue,
            options: [
                .computeCapability(device.computeCapability),
                .useFastMath,
                .disableWarnings,
                .defineMacro("TYPE", as: T.kernelDataType.rawValue)
            ]
        )
        var function: Function!
        device.sync {
            let module = try! Module(ptx: ptx)
            function = module.function(named: String(describing: source))!
            modules[key] = (module, function)
        }
        return function
    }

}
