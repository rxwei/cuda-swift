//
//  Value.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

public struct DeviceValue<Element> : DeviceAddressible {

    var buffer: DeviceValueBuffer<Element>

    private var mutatingBuffer: DeviceValueBuffer<Element> {
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
            return buffer.baseAddress.load()
        }
        set {
            mutatingBuffer.baseAddress.assign(newValue)
        }
    }

    public init(_ initial: Element? = nil) {
        buffer = DeviceValueBuffer()
        initial.flatMap(buffer.baseAddress.assign)
    }

    public init(_ other: DeviceValue<Element>) {
        self = other
    }

    public var unsafePointer: UnsafeDevicePointer<Element> {
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

extension DeviceValue where Element : DeviceCollection {

    public subscript(i: Int) -> Element.Iterator.Element {
        get {
            return value[i]
        }
    }

    public func copyToHost() -> [Element.Element] {
        return value.copyToHost()
    }
    
}

extension DeviceValue where Element : DeviceCollection, Element.Element : DeviceCollection {

    public func copyToHost() -> [[Element.Element.Element]] {
        return value.copyToHost().map{$0.copyToHost()}
    }
    
}
