//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 10/19/16.
//
//

import CCUDARuntime

public struct Device : Equatable {

    public typealias Properties = cudaDeviceProp

    public let index: Int

    public static var count: Int {
        var deviceCount: Int32 = 0
        !!cudaGetDeviceCount(&deviceCount)
        return Int(deviceCount)
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

    public var properties: Properties {
        var prop = Properties()
        !!cudaGetDeviceProperties(&prop, Int32(index))
        return prop
    }

    public var computeCapability: (major: Int, minor: Int) {
        var major: Int32 = 0
        !!cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, Int32(index))
        var minor: Int32 = 0
        !!cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, Int32(index))
        return (major: Int(major), minor: Int(minor))
    }

    public static func ==(lhs: Device, rhs: Device) -> Bool {
        return lhs.index == rhs.index
    }
    
}
