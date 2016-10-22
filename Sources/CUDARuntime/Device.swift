//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 10/19/16.
//
//

import CCUDARuntime

public struct Device : Equatable {

    public struct ComputeCapability : Equatable {
        let major, minor: Int

        public static func ==(lhs: ComputeCapability,
                              rhs: ComputeCapability) -> Bool {
            return lhs.major == rhs.major && lhs.minor == rhs.minor
        }
    }

    public typealias Properties = cudaDeviceProp

    public let index: Int32

    public static var count: Int {
        var deviceCount: Int32 = 0
        !!cudaGetDeviceCount(&deviceCount)
        return Int(deviceCount)
    }

    public static var current: Device {
        var index: Int32 = 0
        !!cudaGetDevice(&index)
        return Device(fromIndex: index)
    }

    internal init(fromIndex index: Int32) {
        self.index = index
    }



    public init?(atIndex index: Int) {
        guard (0..<Device.count).contains(index) else {
            return nil
        }
        self.index = Int32(index)
    }

    public init?(withProperties properties: Properties) {
        var index: Int32 = 0
        var properties = properties
        do {
            try ensureSuccess(cudaChooseDevice(&index, &properties))
            self.index = index
        } catch {
            return nil
        }
    }

    public var properties: Properties {
        var prop = Properties()
        !!cudaGetDeviceProperties(&prop, index)
        return prop
    }

    public var computeCapability: ComputeCapability {
        var major: Int32 = 0
        !!cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, index)
        var minor: Int32 = 0
        !!cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, index)
        return ComputeCapability(major: Int(major), minor: Int(minor))
    }

    public static func ==(lhs: Device, rhs: Device) -> Bool {
        return lhs.index == rhs.index
    }

}
