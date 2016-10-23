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
    let deviceAddress: CUdeviceptr

    /// Initializer for other CUDA Runtime API calls within the module
    public init?(_ deviceAddress: CUdeviceptr) {
        guard deviceAddress != 0 else { return nil }
        self.deviceAddress = deviceAddress
    }

    private init(assumingNonNil deviceAddress: CUdeviceptr) {
        self.deviceAddress = deviceAddress
    }

    public static func allocate(capacity: Int) throws -> UnsafeMutableDevicePointer<Pointee> {
        var address: CUdeviceptr = 0
        try ensureSuccess(cuMemAlloc_v2(&address, capacity))
        return UnsafeMutableDevicePointer(assumingNonNil: address)
    }

    public func deallocate() throws {
        try ensureSuccess(cuMemFree_v2(deviceAddress))
    }

    public var hashValue: Int {
        return deviceAddress.hashValue
    }

    public static func ==(lhs: UnsafeMutableDevicePointer<Pointee>,
                          rhs: UnsafeMutableDevicePointer<Pointee>) -> Bool {
        return lhs.deviceAddress == rhs.deviceAddress
    }

    public func advanced(by n: Int) -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(assumingNonNil:
            deviceAddress.advanced(by: n * MemoryLayout<Pointee>.stride)
        )
    }

    public func distance(to other: UnsafeMutableDevicePointer<Pointee>) -> Int {
        return self.deviceAddress.distance(to: other.deviceAddress)
    }

    public func predecessor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(assumingNonNil:
            deviceAddress.advanced(by: -MemoryLayout<Pointee>.stride)
        )
    }

    public func successor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(assumingNonNil:
            deviceAddress.advanced(by: MemoryLayout<Pointee>.stride)
        )
    }

    /// Pointee **copied** from device
    /// Avoid doing this often because it's expensive
    /// - complexity: O(size)
    public func load() throws -> Pointee {
        var pointee: Pointee?
        try ensureSuccess(
            cuMemcpyDtoH_v2(&pointee, deviceAddress, MemoryLayout<Pointee>.size)
        )
        return pointee!
    }

    public func assign(_ value: Pointee) throws {
        var value = value
        try ensureSuccess(
            cuMemcpyHtoD_v2(deviceAddress, &value, MemoryLayout<Pointee>.stride)
        )
    }

    public func assign(_ value: Pointee, count: Int) throws {
        var value = value
        try ensureSuccess(
            cuMemcpyHtoD_v2(deviceAddress, &value, count * MemoryLayout<Pointee>.stride)
        )
    }

    public func assign<C: Collection>(fromHost elements: C) throws
        where C.Iterator.Element == Pointee, C.IndexDistance == Int {
        try ContiguousArray(elements).withUnsafeBufferPointer { ptr in
            try ensureSuccess(
                cuMemcpyHtoD_v2(self.deviceAddress, ptr.baseAddress,
                                elements.count * MemoryLayout<Pointee>.stride)
            )
        }
    }

    public func assign(fromHost pointer: UnsafePointer<Pointee>, count: Int) throws {
        try ensureSuccess(
            cuMemcpyHtoD_v2(self.deviceAddress, pointer,
                            count * MemoryLayout<Pointee>.stride)
        )
    }

    public func assign(from pointer: UnsafeMutableDevicePointer<Pointee>, count: Int = 1) throws {
        try ensureSuccess(
            cuMemcpyDtoD_v2(self.deviceAddress, pointer.deviceAddress,
                            count * MemoryLayout<Pointee>.stride)
        )
    }

    public func copyBytes(toHost pointer: UnsafeMutablePointer<Pointee>, count: Int) throws {
        try ensureSuccess(
            cuMemcpyDtoH_v2(pointer, deviceAddress,
                            count * MemoryLayout<Pointee>.stride)
        )
    }

    public func withDeviceAddress<Result>
        (body: (CUdeviceptr) throws -> Result) rethrows -> Result {
        return try body(deviceAddress)
    }

    public subscript(i: Int) -> Pointee {
        get {
            return try! self.advanced(by: i).load()
        }
        nonmutating set {
            var newValue = newValue
            cuMemcpyHtoD_v2(advanced(by: i).deviceAddress, &newValue,
                            MemoryLayout<Pointee>.size)
        }
    }
    
}


public extension UnsafeMutablePointer {

    public mutating func assign(fromDevice devPtr: UnsafeMutableDevicePointer<Pointee>, count: Int) throws {
        try devPtr.copyBytes(toHost: self, count: count)
    }
    
}
