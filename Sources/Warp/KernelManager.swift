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

    /// Get shared instance of KernelManager on the specified device.
    ///
    /// - Parameter device: CUDA device
    /// - Returns: kernel manager
    static func shared(on device: Device) -> KernelManager {
        if let manager = instances[device.index] {
            return manager
        }
        let manager = KernelManager(device: device)
        instances[device.index] = manager
        return manager
    }

    let device: Device

    /// Unique hash key for each module
    struct ModuleCacheKey : Equatable, Hashable {
        let type: KernelDataType
        let source: StaticString
        let functor: FloatingPointKernelFunctor?
        let operation: BinaryKernelOperation?

        static func ==(lhs: ModuleCacheKey, rhs: ModuleCacheKey) -> Bool {
            return lhs.type == rhs.type && lhs.source == rhs.source
        }

        var hashValue: Int {
            return type.hashValue ^ source.hashValue &+
                (functor?.hashValue ?? 0) ^ (operation?.hashValue ?? 0)
        }

        init(type: KernelDataType, source: StaticString,
             functor: FloatingPointKernelFunctor? = nil,
             operation: BinaryKernelOperation? = nil) {
            self.type = type
            self.source = source
            self.functor = functor
            self.operation = operation
        }
    }

    /// Cached (compiled and loaded) modules
    fileprivate var modules: [ModuleCacheKey : (Module, Function)] = Dictionary(minimumCapacity: 32)

    /// Initialize an instance on device
    ///
    /// - Parameter device: CUDA device
    init(device: Device) {
        self.device = device
    }

    /// Get a compiled functorial kernel, a.k.a. kernel with a transformation functor
    /// such as `log` and `tanh`.
    ///
    /// - Parameters:
    ///   - source: kernel source to be compiled (if not cached) and loaded
    ///   - functor: 1-place functor for element transformation
    ///   - forType: type of each element
    /// - Returns: kernel function
    func kernel<T: KernelDataProtocol & FloatingPoint>(
        _ source: FunctorialKernelSource, functor: FloatingPointKernelFunctor, forType: T.Type) -> Function {
        /// Get cached function
        let key = ModuleCacheKey(type: T.kernelDataType, source: source.rawValue, functor: functor)
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
                .defineMacro("KERNEL", as: String(describing: source)),
                .defineMacro("TYPE", as: T.kernelDataType.rawValue),
                .defineMacro("FUNC", as: functor.functionName(forType: T.self))
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

    /// Get a compiled operator kernel, a.k.a. kernel with a binary operator
    /// such as `+` and `-`.
    ///
    /// - Parameters:
    ///   - source: kernel source to be compiled (if not cached) and loaded
    ///   - functor: 1-place functor for element transformation
    ///   - forType: type of each element
    /// - Returns: kernel function
    func kernel<T: KernelDataProtocol>(
        _ source: BinaryOperationKernelSource, operation: BinaryKernelOperation, forType: T.Type) -> Function {
        /// Get cached function
        let key = ModuleCacheKey(type: T.kernelDataType, source: source.rawValue, operation: operation)
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
                .defineMacro("KERNEL", as: String(describing: source)),
                .defineMacro("TYPE", as: T.kernelDataType.rawValue),
                .defineMacro("OP", as: operation.operatorSymbol)
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

    /// Get a compiled kernel
    ///
    /// - Parameters:
    ///   - source: kernel source to be compiled (if not cached) and loaded 
    ///   - forType: type of each element
    /// - Returns: kernel function
    func kernel<T: KernelDataProtocol>(_ source: KernelSource, forType: T.Type) -> Function {
        /// Get cached function
        let key = ModuleCacheKey(type: T.kernelDataType, source: source.rawValue)
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
                .defineMacro("KERNEL", as: String(describing: source)),
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
