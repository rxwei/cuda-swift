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
import Foundation

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
        let functor: UnaryOperation?
        let operation: BinaryOperation?

        static func ==(lhs: ModuleCacheKey, rhs: ModuleCacheKey) -> Bool {
            return lhs.type == rhs.type && lhs.source == rhs.source
        }

        var hashValue: Int {
            return type.hashValue ^ source.hashValue &+
                (functor?.hashValue ?? 0) &+ (operation?.hashValue ?? 0)
        }

        init(type: KernelDataType, source: StaticString,
             functor: UnaryOperation? = nil,
             operation: BinaryOperation? = nil) {
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

    /// Print log to stderr
    ///
    /// - Parameter contents: contents of log
    @inline(__always)
    private func log(_ contents: String) {
        contents.withCString { ptr -> () in
            fputs(ptr, stderr)
        }
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
            _ source: FunctorialKernelSource, transformation: UnaryOperation, forType: T.Type) -> Function {
        /// Get cached function
        let key = ModuleCacheKey(type: T.kernelDataType, source: source.rawValue, functor: transformation)
        if let (_, function) = modules[key] {
            return function
        }
        /// If not cached, compile using NVRTC
        log("Loading CUDA kernel \'\(source) \(transformation)\' for \(T.self)...\n")
        let ptx = try! Compiler.compile(
            source.rawValue,
            options: [
                .computeCapability(device.computeCapability),
                .useFastMath,
                .disableWarnings,
                .defineMacro("KERNEL", as: String(describing: source)),
                .defineMacro("TYPE", as: T.kernelDataType.rawValue),
                .defineMacro("FUNC", as: transformation.functionName(forType: T.self))
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
    ///   - source: kernel source to be compiled and loaded, if not cached
    ///   - functor: 1-place functor for element transformation
    ///   - forType: type of each element
    /// - Returns: kernel function
    func kernel<T: KernelDataProtocol>(
            _ source: BinaryOperationKernelSource, operation: BinaryOperation, forType: T.Type) -> Function {
        /// Get cached function
        let key = ModuleCacheKey(type: T.kernelDataType, source: source.rawValue, operation: operation)
        if let (_, function) = modules[key] {
            return function
        }
        /// If not cached, compile using NVRTC
        log("Loading CUDA kernel \'\(source) \(operation)\' for \(T.self)...\n")
        let ptx = try! Compiler.compile(
            source.rawValue,
            options: [
                .computeCapability(device.computeCapability),
                .useFastMath,
                .disableWarnings,
                .defineMacro("KERNEL", as: String(describing: source)),
                .defineMacro("TYPE", as: T.kernelDataType.rawValue),
                .defineMacro("OP(_x_, _y_)", as: operation.macro)
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
        log("Loading CUDA kernel \'\(source)\' for \(T.self)...\n")
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
