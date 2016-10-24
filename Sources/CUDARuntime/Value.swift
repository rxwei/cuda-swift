//
//  Value.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

final class DeviceValueBuffer<Wrapped> {

    let address: UnsafeMutableDevicePointer<Wrapped>

    init(_ initial: Wrapped? = nil) {
        address = UnsafeMutableDevicePointer.allocate(capacity: 1)
        if let initial = initial {
            address.assign(initial)
        }
    }

    init(_ other: DeviceValueBuffer<Wrapped>) {
        address = UnsafeMutableDevicePointer.allocate(capacity: 1)
        address.assign(from: other.address)
    }

    deinit {
        address.deallocate()
    }

}

public struct DeviceValue<Wrapped> {

    private var buffer: DeviceValueBuffer<Wrapped>

    private var cowBuffer: DeviceValueBuffer<Wrapped> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) {
                buffer = DeviceValueBuffer(buffer)
            }
            return buffer
        }
    }

    public var value: Wrapped {
        get {
            return buffer.address.load()
        }
        mutating set {
            cowBuffer.address.assign(newValue)
        }
    }

    public init(_ initial: Wrapped? = nil) {
        buffer = DeviceValueBuffer(initial)
    }

    public init(_ other: DeviceValue<Wrapped>) {
        self = other
    }

    /// - note: This should be @inline(__always). But it crashes Swift SIL
    /// verfication as of Swift 3.0.1-PREVIEW-3
    /// [SR-3030](https://bugs.swift.org/browse/SR-3030)
    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (UnsafeMutableDevicePointer<Wrapped>) throws -> Result) rethrows -> Result {
        return try body(cowBuffer.address)
    }

}

