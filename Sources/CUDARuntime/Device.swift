//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 10/19/16.
//
//

import CCUDARuntime
import struct CUDADriver.Device
@_exported import struct CUDADriver.ComputeCapability

public struct Device : Equatable {

    public typealias Properties = cudaDeviceProp

    public let index: Int

    public static var count: Int {
        var deviceCount: Int32 = 0
        !!cudaGetDeviceCount(&deviceCount)
        return Int(deviceCount)
    }

    public static var all: [Device] = (0..<Int32(count)).map { index in
        Device(assumingIndex: index)
    }

    public static var current: Device {
        get {
            var index: Int32 = 0
            !!cudaGetDevice(&index)
            return Device(assumingIndex: index)
        }
        set {
            !!cudaSetDevice(Int32(newValue.index))
        }
    }

    /// Unsafely assume the index handle to be the device index
    internal init(assumingIndex index: Int32) {
        self.index = Int(index)
    }

    public init?(atIndex index: Int) {
        guard index >= 0 && index < Device.count else {
            return nil
        }
        self.index = index
    }

    public init?(withProperties properties: Properties) {
        var index: Int32 = 0
        var properties = properties
        do {
            try ensureSuccess(cudaChooseDevice(&index, &properties))
            self.index = Int(index)
        } catch {
            return nil
        }
    }

    public func sync(_ execute: () throws -> ()) rethrows {
        let contexualDevice = Device.current
        /// Synchronously execute directly if it's the same device
        if self == contexualDevice {
            Device.synchronize()
            try execute()
            Device.synchronize()
        }
        /// Otherwise, switch back to the previous device after executing
        else {
            Device.current = self
            Device.synchronize()
            try execute()
            Device.synchronize()
            Device.current = contexualDevice
        }
    }

    public static func synchronize() {
        !!cudaDeviceSynchronize()
    }

    public var properties: Properties {
        var prop = Properties()
        !!cudaGetDeviceProperties(&prop, Int32(index))
        return prop
    }

    public var computeCapability: ComputeCapability {
        var major: Int32 = 0
        !!cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, Int32(index))
        var minor: Int32 = 0
        !!cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, Int32(index))
        return ComputeCapability(major: Int(major), minor: Int(minor))
    }

    public static func ==(lhs: Device, rhs: Device) -> Bool {
        return lhs.index == rhs.index
    }

    public init(_ driverDevice: CUDADriver.Device) {
        self.init(assumingIndex: Int32(driverDevice.index))
    }

    /// Convert to driver device
    public var driverDevice: CUDADriver.Device {
        return CUDADriver.Device(atIndex: index)!
    }
    
}

/// Conversion from runtime device
public extension CUDADriver.Device {
    public init(_ device: CUDARuntime.Device) {
        self = device.driverDevice
    }

    public var runtimeDevice: CUDARuntime.Device {
        return CUDARuntime.Device(self)
    }
}
