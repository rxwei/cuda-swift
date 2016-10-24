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

    let handle: CUdevice

    init(_ handle: CUdevice) {
        self.handle = handle
    }

    public var name: String {
        var name: [CChar] = Array(repeating: 0, count: 32)
        cuDeviceGetName(&name, 32, handle)
        return String(cString: name)
    }

    public var computeCapability: (major: Int, minor: Int) {
        var major: Int32 = 0, minor: Int32 = 0
        cuDeviceComputeCapability(&major, &minor, handle)
        return (major: Int(major), minor: Int(minor))
    }

    public var properties: Properties {
        var props: CUdevprop = CUdevprop()
        cuDeviceGetProperties(&props, handle)
        return props
    }

    public var pciBusID: String {
        var id: [CChar] = Array(repeating: 0, count: 32)
        cuDeviceGetPCIBusId(&id, 32, handle)
        return String(cString: id)
    }

    public func makeContext() -> Context {
        var ctxHandle: CUcontext?
        !!cuCtxCreate_v2(&ctxHandle, 0, handle)
        return Context(binding: ctxHandle!)
    }

    public static func ==(lhs: Device, rhs: Device) -> Bool {
        return lhs.handle == rhs.handle
    }

    public init?(atIndex index: Int) {
        var handle: CUdevice = 0
        do {
            try ensureSuccess(cuDeviceGet(&handle, Int32(index)))
            self.handle = handle
        } catch {
            return nil
        }
    }

    public func withUnsafeHandle<Result>
        (_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

    public static var count: Int {
        var deviceCount: Int32 = 0
        cuDeviceGetCount(&deviceCount)
        return Int(deviceCount)
    }

    public static var main: Device = Device(atIndex: 0)!

}
