//
//  Memory.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public enum SharedMemoryBankSize: UInt32 {
    case `default` = 0x00
    case fourBytes = 0x01
    case eightBytes = 0x02
}

public struct UnsafeMutableDevicePointer<Pointee> : Equatable, Hashable, Strideable {

    public typealias Stride = Int

    /// Raw address on CUDA device
    let deviceAddressHandle: CUdeviceptr

    private var deviceAddress: UnsafePointer<Pointee> {
        /// Safe since deviceAddressHandle is never 0!
        return UnsafePointer(bitPattern: UInt(deviceAddressHandle))!
    }

    /// Initializer for other CUDA Runtime API calls within the module
    public init?(_ deviceAddressHandle: CUdeviceptr) {
        guard deviceAddressHandle != 0 else { return nil }
        self.deviceAddressHandle = deviceAddressHandle
    }

    private init(assumingNonNil deviceAddress: CUdeviceptr) {
        self.deviceAddressHandle = deviceAddress
    }

    public static func allocate(capacity: Int) throws -> UnsafeMutableDevicePointer<Pointee> {
        var address: CUdeviceptr = 0
        try ensureSuccess(cuMemAlloc_v2(&address, capacity * MemoryLayout<Pointee>.stride))
        return UnsafeMutableDevicePointer(assumingNonNil: address)
    }

    public func deallocate() throws {
        try ensureSuccess(cuMemFree_v2(deviceAddressHandle))
    }

    public var hashValue: Int {
        return deviceAddressHandle.hashValue
    }

    public static func ==(lhs: UnsafeMutableDevicePointer<Pointee>,
                          rhs: UnsafeMutableDevicePointer<Pointee>) -> Bool {
        return lhs.deviceAddressHandle == rhs.deviceAddressHandle
    }

    public func advanced(by n: Int) -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(assumingNonNil:
            deviceAddressHandle.advanced(by: n * MemoryLayout<Pointee>.stride)
        )
    }

    public func distance(to other: UnsafeMutableDevicePointer<Pointee>) -> Int {
        return self.deviceAddressHandle.distance(to: other.deviceAddressHandle)
    }

    public func predecessor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(assumingNonNil:
            deviceAddressHandle.advanced(by: -MemoryLayout<Pointee>.stride)
        )
    }

    public func successor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(assumingNonNil:
            deviceAddressHandle.advanced(by: MemoryLayout<Pointee>.stride)
        )
    }

    /// Pointee **copied** from device
    /// Avoid doing this often because it's expensive
    /// - complexity: O(size)
    public func load() throws -> Pointee {
        var pointee: Pointee?
        try ensureSuccess(
            cuMemcpyDtoH_v2(&pointee, deviceAddressHandle, MemoryLayout<Pointee>.size)
        )
        return pointee!
    }

    public func assign(_ value: Pointee) throws {
        var value = value
        try ensureSuccess(
            cuMemcpyHtoD_v2(deviceAddressHandle, &value, MemoryLayout<Pointee>.size)
        )
    }

    public func assign(_ value: Pointee, count: Int) throws {
        var value = value
        try ensureSuccess(
            cuMemcpyHtoD_v2(deviceAddressHandle, &value, count * MemoryLayout<Pointee>.stride)
        )
    }

    public func assign<C: Collection>(fromHost elements: C) throws
        where C.Iterator.Element == Pointee, C.IndexDistance == Int {
        try ContiguousArray(elements).withUnsafeBufferPointer { ptr in
            try ensureSuccess(
                cuMemcpyHtoD_v2(self.deviceAddressHandle, ptr.baseAddress,
                                elements.count * MemoryLayout<Pointee>.stride)
            )
        }
    }

    public func assign(fromHost pointer: UnsafePointer<Pointee>) throws {
        try ensureSuccess(
            cuMemcpyHtoD_v2(self.deviceAddressHandle, pointer,
                            MemoryLayout<Pointee>.size)
        )
    }

    public func assign(fromHost pointer: UnsafePointer<Pointee>, count: Int) throws {
        try ensureSuccess(
            cuMemcpyHtoD_v2(self.deviceAddressHandle, pointer,
                            count * MemoryLayout<Pointee>.stride)
        )
    }

    public func assign(from pointer: UnsafeMutableDevicePointer<Pointee>) throws {
        try ensureSuccess(
            cuMemcpyDtoD_v2(self.deviceAddressHandle, pointer.deviceAddressHandle,
                            MemoryLayout<Pointee>.stride)
        )
    }

    public func assign(from pointer: UnsafeMutableDevicePointer<Pointee>, count: Int) throws {
        try ensureSuccess(
            cuMemcpyDtoD_v2(self.deviceAddressHandle, pointer.deviceAddressHandle,
                            count * MemoryLayout<Pointee>.stride)
        )
    }

    public func copyBytes(toHost pointer: UnsafeMutablePointer<Pointee>, count: Int) throws {
        try ensureSuccess(
            cuMemcpyDtoH_v2(pointer, deviceAddressHandle,
                            count * MemoryLayout<Pointee>.stride)
        )
    }

    public func withDeviceAddressHandle<Result>
        (body: (CUdeviceptr) throws -> Result) rethrows -> Result {
        return try body(deviceAddressHandle)
    }

    public func withDeviceAddress<Result>
        (body: (UnsafePointer<Pointee>) throws -> Result) rethrows -> Result {
        return try body(deviceAddress)
    }

    public subscript(i: Int) -> Pointee {
        get {
            return try! self.advanced(by: i).load()
        }
        nonmutating set {
            var newValue = newValue
            cuMemcpyHtoD_v2(advanced(by: i).deviceAddressHandle, &newValue,
                            MemoryLayout<Pointee>.size)
        }
    }
    
}

public extension UnsafeMutablePointer {

    public mutating func assign(fromDevice devPtr: UnsafeMutableDevicePointer<Pointee>, count: Int) throws {
        try devPtr.copyBytes(toHost: self, count: count)
    }
    
}
