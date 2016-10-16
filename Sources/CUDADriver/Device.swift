//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public final class DeviceManager {

    public static var deviceCount: Int {
        var deviceCount: Int32 = 0
        cuDeviceGetCount(&deviceCount)
        return Int(deviceCount)
    }

    public static func device(at index: Int) -> Device? {
        var handle: CUdevice = 0
        guard cuDeviceGet(&handle, Int32(index)) == CUDA_SUCCESS else {
            return nil
        }
        return Device(deviceHandle: handle)
    }

}


public struct Device {

    public struct ComputeCapability {
        let major, minor: Int
    }

    public typealias Properties = CUdevprop

    public static var `default`: Device {
        return DeviceManager.device(at: 0)!
    }

    let handle: CUdevice

    init(deviceHandle: CUdevice) {
        self.handle = deviceHandle
    }

    public var name: String {
        var name: [CChar] = Array(repeating: 0, count: 32)
        cuDeviceGetName(&name, 32, handle)
        return String(cString: name)
    }

    public var computeCapability: ComputeCapability {
        var major: Int32 = 0, minor: Int32 = 0
        cuDeviceComputeCapability(&major, &minor, handle)
        return ComputeCapability(major: Int(major), minor: Int(minor))
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

}
