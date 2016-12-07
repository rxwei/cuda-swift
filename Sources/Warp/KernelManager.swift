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

    /// Current CUDA device
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
    fileprivate var modules: [ModuleCacheKey : (Module, Function)] = Dictionary(minimumCapacity: 128)

    /// Initialize an instance on device
    ///
    /// - Parameter device: CUDA device
    init(device: Device) {
        self.device = device
    }

    /// Common compile options
    private lazy var commonCompileOptions: [CompileOption] = [
        .computeCapability(self.device.computeCapability),
        .useFastMath,
        .disableWarnings,
        .defineMacro("IDX(_X_)", as: "long long (_X_)=blockIdx.x*blockDim.x+threadIdx.x")
    ]

    /// Get a compiled functorial kernel, a.k.a. kernel with a transformation functor
    /// such as `log` and `tanh`.
    /// - Note: This supports only floating point types, because unary math functions
    /// such as `sin` and `cos` are available only for floating point.
    ///
    /// - Parameters:
    ///   - source: kernel source to be compiled (if not cached) and loaded
    ///   - functor: 1-place functor for element transformation
    ///   - forType: type of each element
    /// - Returns: kernel function
    func kernel<T: KernelDataProtocol & FloatingPoint>(_ source: FunctorialKernelSource,
                                                       transformation: UnaryOperation,
                                                       forType: T.Type) -> Function {
        /// Get cached function
        let key = ModuleCacheKey(type: T.kernelDataType, source: source.rawValue, functor: transformation)
        if let (_, function) = modules[key] {
            return function
        }
        /// If not cached, compile using NVRTC
        let (module, function) = makeKernel(source, extraOptions: [
            .defineMacro("TYPE", as: T.kernelDataType.rawValue),
            .defineMacro("FUNC", as: transformation.source(forType: T.self))
        ])
        modules[key] = (module, function)
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
    func kernel<T: KernelDataProtocol>(_ source: BinaryOperationKernelSource,
                                       operation: BinaryOperation,
                                       forType: T.Type) -> Function {
        /// Get cached function
        let key = ModuleCacheKey(type: T.kernelDataType, source: source.rawValue, operation: operation)
        if let (_, function) = modules[key] {
            return function
        }
        /// If not cached, compile using NVRTC
        log("Loading CUDA kernel \'\(source) \(operation)\' for \(T.self)...\n")
        let (module, function) = makeKernel(source, extraOptions: [
            .defineMacro("TYPE", as: T.kernelDataType.rawValue),
            .defineMacro("OP(_x_, _y_)", as: operation.source)
        ])
        modules[key] = (module, function)
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
        let (module, function) = makeKernel(source, extraOptions: [
            .defineMacro("TYPE", as: T.kernelDataType.rawValue)
        ])
        modules[key] = (module, function)
        return function
    }

    /// Compile kernel source and load as a module
    ///
    /// - Parameters:
    ///   - source: source to compile
    ///   - extraOptions: options other than `commonCompileOptions` 
    ///     e.g. type macro definitions
    /// - Returns: loaded module and function reference
    func makeKernel<Source : SourceHashable>(_ source: Source,
                                             extraOptions: [CompileOption]) -> (Module, Function)
        where Source : RawRepresentable, Source.RawValue == StaticString {
        let ptx = try! Compiler.compile(
            source.rawValue,
            options: commonCompileOptions +
                [ .defineMacro("KERNEL", as: "extern \"C\" __global__ void \(source)") ] +
                extraOptions
        )
        var module: Module!
        var function: Function!
        device.sync {
            module = try! Module(ptx: ptx)
            function = module.function(named: String(describing: source))!
        }
        return (module, function)
    }

}

/// Print log to stderr
///
/// - Parameter contents: contents of log
func log(_ contents: String) {
    contents.withCString { ptr -> () in
        fputs(ptr, stderr)
    }
}
