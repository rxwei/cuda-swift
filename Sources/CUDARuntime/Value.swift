//
//  Value.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

/// Tests broken

fileprivate final class DeviceValueBuffer<Wrapped> {

    let address: UnsafeMutableDevicePointer<Wrapped>

    init(_ initial: Wrapped? = nil) {
        address = UnsafeMutableDevicePointer.allocate(capacity: MemoryLayout<Wrapped>.size)
        if let initial = initial {
            address.assign(initial)
        }
    }

    init(_ other: DeviceValueBuffer<Wrapped>) {
        address = UnsafeMutableDevicePointer.allocate(capacity: MemoryLayout<Wrapped>.size)
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
        set {
            cowBuffer.address.assign(value)
        }
    }

    public init(_ initial: Wrapped? = nil) {
        buffer = DeviceValueBuffer(initial)
    }

}

