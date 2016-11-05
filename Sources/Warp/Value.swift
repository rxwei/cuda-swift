//
//  Value.swift
//  Warp
//
//  Created by Richard Wei on 10/23/16.
//
//

import CUDARuntime

public struct DeviceValue<Element> : DeviceAddressible {

    var buffer: DeviceValueBuffer<Element>

    var mutatingBuffer: DeviceValueBuffer<Element> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) || buffer.owner != nil {
                buffer = DeviceValueBuffer(buffer)
            }
            return buffer
        }
    }

    init(buffer: DeviceValueBuffer<Element>) {
        self.buffer = buffer
    }

    public var value: Element {
        get {
            return buffer.value
        }
        set {
            mutatingBuffer.value = newValue
        }
    }

    public init(_ initialValue: Element? = nil) {
        buffer = DeviceValueBuffer()
        if let initialValue = initialValue {
            buffer.value = initialValue
        }
    }

    public init(_ other: DeviceValue<Element>) {
        self = other
    }

    var unsafePointer: UnsafeDevicePointer<Element> {
        return UnsafeDevicePointer(buffer.baseAddress)
    }

    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try body(UnsafeDevicePointer(buffer.baseAddress))
    }

    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result {
        var baseAddress = mutatingBuffer.baseAddress
        return try body(&baseAddress)
    }

}

extension DeviceValue where Element : MutableDeviceCollection {

    public subscript(i: Int) -> Element.Iterator.Element {
        set {
            value[i] = newValue
        }
        get {
            return value[i]
        }
    }

}

extension DeviceValue where Element : DeviceCollection {

    public var hostArray: [Element.Element] {
        return value.hostArray
    }
    
}

extension DeviceValue where Element : DeviceCollection, Element.Element : DeviceCollection {

    public var hostArray: [[Element.Element.Element]] {
        return value.hostArray.map{$0.hostArray}
    }
    
}
