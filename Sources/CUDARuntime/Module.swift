//
//  Module.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

/// This is a bridge from CUDADriver
import CCUDARuntime
@_exported import struct CUDADriver.PTX
@_exported import struct CUDADriver.Function
@_exported import struct CUDADriver.ArgumentList
@_exported import struct CUDADriver.GridSize
@_exported import struct CUDADriver.BlockSize
import enum CUDADriver.Driver
import struct CUDADriver.Device
import struct CUDADriver.UnsafeMutableDevicePointer
import struct CUDADriver.Context
import class CUDADriver.Module
import class CUDADriver.Stream

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

    deinit {
        context.end()
    }

}

public extension CUDADriver.ArgumentList {

    public mutating func append<DeviceType: DeviceAddressible>(_ argument: inout DeviceType) {
        argument.withUnsafeMutableDevicePointer { devPtr -> () in
            devPtr.withMutableDeviceAddress { addr -> () in
                self.append(addr)
                return () /// Compiler bug
            }
        }
    }

}

public extension CUDADriver.Function {

    public func launch(with arguments: inout ArgumentList,
                       gridSize: GridSize, blockSize: BlockSize, stream: CUDARuntime.Stream? = nil) throws {
        try launch(with: &arguments,
                   gridSize: gridSize,
                   blockSize: blockSize,
                   stream: stream.flatMap { CUDADriver.Stream(unsafelyReferencing: $0.handle) })
    }

    public func launch(with arguments: inout ArgumentList,
                       blockCount: Int, threadCount: Int,
                       memory: Int = 0, stream: CUDARuntime.Stream? = nil) throws {
        try launch(with: &arguments,
                   gridSize: GridSize(blockCount),
                   blockSize: BlockSize(threadCount: threadCount, memory: memory),
                   stream: stream.flatMap { CUDADriver.Stream(unsafelyReferencing: $0.handle) })
    }
    
}

infix operator <<< : CUDAKernelPrecedence
infix operator >>> : CUDAKernelPrecedence

/// CUDA-like <<<>>> operator
public extension Function {

    public static func <<<(lhs: Function, rhs: (Int, Int)) -> (inout ArgumentList) throws -> () {
        return {
            try lhs.launch(with: &$0, blockCount: rhs.0, threadCount: rhs.1)
        }
    }

    public static func <<<(lhs: Function, rhs: (Int, Int, Int)) -> (inout ArgumentList) throws -> () {
        return {
            try lhs.launch(with: &$0, blockCount: rhs.0, threadCount: rhs.1, memory: rhs.2)
        }
    }

    public static func <<<(lhs: Function, rhs: (Int, Int, Int, CUDARuntime.Stream)) -> (inout ArgumentList) throws -> () {
        return {
            try lhs.launch(with: &$0, blockCount: rhs.0, threadCount: rhs.1, memory: rhs.2, stream: rhs.3)
        }
    }

}

public func >>>(lhs: (inout ArgumentList) throws -> (),
                rhs: (inout ArgumentList)) rethrows {
    try lhs(&rhs)
}
