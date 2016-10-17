//
//  Device.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public final class Device {

    public struct ComputeCapability : Equatable {
        let major, minor: Int

        public static func ==(lhs: ComputeCapability,
                              rhs: ComputeCapability) -> Bool {
            return lhs.major == rhs.major && lhs.minor == rhs.minor
        }
    }

    public typealias Properties = CUdevprop

    let handle: CUdevice

    fileprivate init(handle: CUdevice) {
        self.handle = handle
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

public final class DeviceManager {

    public static let shared = try! DeviceManager()

    public lazy var devices: [Device] = {
        (0..<self.deviceCount).flatMap { try? self.device(at:$0) }
    }()

    internal func device(withHandle handle: CUdevice) -> Device {
        /// It's a bad assumption to iterate but it's a very tiny search space
        return devices.first(where: {$0.handle == handle})!
    }

    fileprivate var deviceCount: Int {
        var deviceCount: Int32 = 0
        cuDeviceGetCount(&deviceCount)
        return Int(deviceCount)
    }

    fileprivate func device(at index: Int) throws -> Device {
        var handle: CUdevice = 0
        try ensureSuccess(cuDeviceGet(&handle, Int32(index)))
        return Device(handle: handle)
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
