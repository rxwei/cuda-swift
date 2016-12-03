//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public struct Device : Equatable, CHandleCarrier {

    public typealias Handle = CUdevice

    public typealias Properties = CUdevprop

    /// First device
    public static var main: Device = Device(atIndex: 0)!

    /// All available devices
    public static var all: [Device] = (0..<count).map {
        Device(atIndex: $0)!
    }

    let handle: CUdevice
    
    public var index: Int {
        return Int(handle)
    }

    init(_ handle: CUdevice) {
        self.handle = handle
    }

    public static var count: Int {
        var deviceCount: Int32 = 0
        !!cuDeviceGetCount(&deviceCount)
        return Int(deviceCount)
    }

    public static func ==(lhs: Device, rhs: Device) -> Bool {
        return lhs.handle == rhs.handle
    }

    public var name: String {
        var name: [CChar] = Array(repeating: 0, count: 32)
        !!cuDeviceGetName(&name, 32, handle)
        return String(cString: name)
    }

    public var computeCapability: ComputeCapability {
        var major: Int32 = 0, minor: Int32 = 0
        !!cuDeviceComputeCapability(&major, &minor, handle)
        return ComputeCapability(major: Int(major), minor: Int(minor))
    }

    public var properties: Properties {
        var props: CUdevprop = CUdevprop()
        !!cuDeviceGetProperties(&props, handle)
        return props
    }

    public var pciBusIdentifier: String {
        var id: [CChar] = Array(repeating: 0, count: 32)
        !!cuDeviceGetPCIBusId(&id, 32, handle)
        return String(cString: id)
    }

    /// Create a context and execute the body synchronously.
    /// The lifetime of the context = the lifetype of the body
    /// - Parameter execute: closure with the new context
    /// - Throws: whatever the closure throws
    public func sync(_ execute: (Context) throws -> ()) rethrows {
        var context = Context.begin(on: self)
        try execute(context)
        Context.synchronize()
        context.end()
    }

    public func makeContext() -> Context {
        return Context.begin(on: self)
    }

    public init?(atIndex index: Int) {
        var handle: CUdevice = 0
        guard cuDeviceGet(&handle, Int32(index)) == CUDA_SUCCESS else {
            return nil
        }
        self.handle = handle
    }

    public func withUnsafeHandle<Result>
        (_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

}

public struct ComputeCapability {

    public let major, minor: Int

    public init(major: Int, minor: Int) {
        self.major = major
        self.minor = minor
    }
    
}
