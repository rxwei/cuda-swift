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

    fileprivate var modules: [KernelSource : Module] = [:]

    init(device: Device) {
        self.device = device
    }

    func launchKernel<T: KernelDataProtocol>
        (from source: KernelSource, forType type: T.Type,
         arguments: [KernelArgument], gridSize: GridSize, blockSize: BlockSize, stream: Stream? = nil) throws {
        try device.withContext { context in
            let module = try modules[source] ??
                Module(source: source.rawValue,
                       compileOptions: [.defineMacro("TYPE", as: T.kernelTypeName)])
            let function = module.function(named: String(describing: source))!
            try function.launch(with: arguments, gridSize: gridSize, blockSize: blockSize, stream: stream)
        }
    }

}
