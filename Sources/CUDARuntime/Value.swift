//
//  Value.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

final class DeviceValueBuffer<Wrapped> {

    let address: UnsafeMutableDevicePointer<Wrapped>

    let owning: Bool

    init(_ initial: Wrapped? = nil) {
        owning = true
        address = UnsafeMutableDevicePointer.allocate(capacity: 1)
        if let initial = initial {
            address.assign(initial)
        }
    }

    init(unownedPointer: UnsafeMutableDevicePointer<Wrapped>) {
        owning = false
        address = unownedPointer
    }

    init(_ other: DeviceValueBuffer<Wrapped>) {
        owning = true
        address = UnsafeMutableDevicePointer.allocate(capacity: 1)
        address.assign(from: other.address)
    }

    deinit {
        if owning {
            address.deallocate()
        }
    }

}

public struct DeviceValue<Wrapped> {

    private var buffer: DeviceValueBuffer<Wrapped>

    private var cowBuffer: DeviceValueBuffer<Wrapped> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) || !buffer.owning {
                buffer = DeviceValueBuffer(buffer)
            }
            return buffer
        }
    }

    public var value: Wrapped {
        nonmutating get {
            return buffer.address.load()
        }
        mutating set {
            cowBuffer.address.assign(newValue)
        }
    }

    public init(_ initial: Wrapped? = nil) {
        buffer = DeviceValueBuffer(initial)
    }

    internal init(unownedReference address: UnsafeMutableDevicePointer<Wrapped>) {
        buffer = DeviceValueBuffer(unownedPointer: address)
    }

    public init(_ other: DeviceValue<Wrapped>) {
        self = other
    }

    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (UnsafeMutableDevicePointer<Wrapped>) throws -> Result) rethrows -> Result {
        return try body(cowBuffer.address)
    }

    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Wrapped>) throws -> Result) rethrows -> Result {
        return try body(UnsafeDevicePointer(buffer.address))
    }

}
