//
//  Value.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

public struct DeviceValue<Wrapped> {

    private var buffer: ManagedDeviceBuffer<Wrapped>
    private var owning = true

    private var mutatingBuffer: ManagedDeviceBuffer<Wrapped> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) || !owning {
                buffer = ManagedDeviceBuffer(buffer)
            }
            return buffer
        }
    }

    init(buffer: ManagedDeviceBuffer<Wrapped>) {
        self.buffer = buffer
        self.owning = false
    }

    public var value: Wrapped {
        get {
            return buffer.baseAddress.load()
        }
        set {
            mutatingBuffer.baseAddress.assign(newValue)
        }
    }

    public init(_ initial: Wrapped? = nil) {
        buffer = ManagedDeviceBuffer(capacity: 1)
        initial.flatMap(buffer.baseAddress.assign)
    }

    public init(_ other: DeviceValue<Wrapped>) {
        self = other
    }

    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (UnsafeMutableDevicePointer<Wrapped>) throws -> Result) rethrows -> Result {
        return try body(mutatingBuffer.baseAddress)
    }

    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Wrapped>) throws -> Result) rethrows -> Result {
        return try body(UnsafeDevicePointer(buffer.baseAddress))
    }

}
