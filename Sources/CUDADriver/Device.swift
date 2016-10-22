//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public struct Device : Equatable {

    public struct ComputeCapability : Equatable {
        let major, minor: Int

        public static func ==(lhs: ComputeCapability,
                              rhs: ComputeCapability) -> Bool {
            return lhs.major == rhs.major && lhs.minor == rhs.minor
        }
    }

    public typealias Properties = CUdevprop

    public let unsafeHandle: CUdevice

    internal init(from handle: CUdevice) {
        self.unsafeHandle = handle
    }

    public var name: String {
        var name: [CChar] = Array(repeating: 0, count: 32)
        cuDeviceGetName(&name, 32, unsafeHandle)
        return String(cString: name)
    }

    public var computeCapability: ComputeCapability {
        var major: Int32 = 0, minor: Int32 = 0
        cuDeviceComputeCapability(&major, &minor, unsafeHandle)
        return ComputeCapability(major: Int(major), minor: Int(minor))
    }

    public var properties: Properties {
        var props: CUdevprop = CUdevprop()
        cuDeviceGetProperties(&props, unsafeHandle)
        return props
    }

    public var pciBusID: String {
        var id: [CChar] = Array(repeating: 0, count: 32)
        cuDeviceGetPCIBusId(&id, 32, unsafeHandle)
        return String(cString: id)
    }

    public static func ==(lhs: Device, rhs: Device) -> Bool {
        return lhs.unsafeHandle == rhs.unsafeHandle
    }

}

public final class DeviceManager {

    public static let shared = try! DeviceManager()

    public lazy var devices: [Device] = {
        (0..<self.deviceCount).flatMap { try? self.device(at: $0) }
    }()

    fileprivate var deviceCount: Int {
        var deviceCount: Int32 = 0
        cuDeviceGetCount(&deviceCount)
        return Int(deviceCount)
    }

    fileprivate func device(at index: Int) throws -> Device {
        var handle: CUdevice = 0
        try ensureSuccess(cuDeviceGet(&handle, Int32(index)))
        return Device(from: handle)
    }

    private init() throws {
        try ensureSuccess(cuInit(0))
    }

}

public extension Device {

    public static var `default`: Device {
        guard let first = try? DeviceManager.shared.device(at: 0) else {
            fatalError("No CUDA devices available")
        }
        return first
    }
    
}
