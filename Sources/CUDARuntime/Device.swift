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

    internal init(index: Int32) {
        self.index = index
    }

    public var properties: Properties {
        var prop = Properties()
        !!cudaGetDeviceProperties(&prop, index)
        return prop
    }

    public static func ==(lhs: Device, rhs: Device) -> Bool {
        return lhs.index == rhs.index
    }

}

public final class DeviceManager {

    public static let shared = DeviceManager()

    
    fileprivate var deviceCount: Int {
        var deviceCount: Int32 = 0
        !!cudaGetDeviceCount(&deviceCount)
        return Int(deviceCount)
    }

    public func device(with properties: Device.Properties) throws -> Device {
        var index: Int32 = 0
        var properties = properties
        try ensureSuccess(cudaChooseDevice(&index, &properties))
        return Device(index: index)
    }

}
