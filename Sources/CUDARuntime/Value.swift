//
//  Value.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

public struct DeviceValue<Wrapped> {

    var buffer: DeviceValueBuffer<Wrapped>

    private var mutatingBuffer: DeviceValueBuffer<Wrapped> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) || buffer.owner != nil {
                buffer = DeviceValueBuffer(buffer)
            }
            return buffer
        }
    }

    init(buffer: DeviceValueBuffer<Wrapped>) {
        self.buffer = buffer
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
        buffer = DeviceValueBuffer()
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

extension DeviceValue where Wrapped : DeviceCollection {

    public subscript(i: Int) -> Wrapped.Iterator.Element {
        get {
            return value[i]
        }
    }

    public func copyToHost() -> [Wrapped.Element] {
        return value.copyToHost()
    }
    
}

extension DeviceValue where Wrapped : DeviceCollection, Wrapped.Element : DeviceCollection {

    public func copyToHost() -> [[Wrapped.Element.Element]] {
        return value.copyToHost().map{$0.copyToHost()}
    }
    
}
