//
//  KernelManager.swift
//  CUDA
//
//  Created by Richard Wei on 11/3/16.
//
//

import CUDADriver
import NVRTC

final class KernelManager {

    fileprivate static var instances: [Int : KernelManager] = [:]

    static func global(on device: Device) -> KernelManager {
        if let manager = instances[device.index] {
            return manager
        }
        let manager = KernelManager(device: device)
        instances[device.index] = manager
        return manager
    }

    let device: Device
    var context: Context?

    fileprivate var modules: [String : [KernelSource : Module]] = Dictionary(minimumCapacity: 16)

    init(device: Device) {
        self.device = device
    }

    deinit {
        context?.end()
    }

    func launchKernel<T: KernelDataProtocol>(_ source: KernelSource, forType type: T.Type,
                      arguments: [KernelArgument], blockCount: Int, threadCount: Int,
                      memory: Int = 0, stream: Stream? = nil) {
        /// Check and add entry for type T
        let cTypeName = T.kernelTypeName
        if !modules.keys.contains(T.kernelTypeName) {
            modules[cTypeName] = Dictionary(minimumCapacity: 32)
        }

        /// Push context
        if let context = context {
            context.pushToThread()
        } else {
            context = Context.begin(on: device)
        }

        /// Load module from cache. If it's not cached, compile kernel from source
        let module: Module
        if let cachedModule = modules[cTypeName]![source] {
            module = cachedModule
        } else {
            /// Compile using NVRTC
            module = try! Module(
                source: source.rawValue,
                compileOptions: [
                    .computeCapability(device.computeCapability),
                    .useFastMath,
                    .disableWarnings,
                    .defineMacro("TYPE", as: T.kernelTypeName)
                ]
            )
            /// Cache it
            modules[cTypeName]![source] = module
        }

        /// Launch function
        let function = module.function(named: String(describing: source))!
        try! function<<<(blockCount, threadCount, memory, stream)>>>(arguments)
        try! Context.synchronize()

        /// Pop context
        Context.popFromThread()
    }

}
