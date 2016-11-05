//
//  Array.swift
//  Warp
//
//  Created by Richard Wei on 10/19/16.
//
//

import CUDARuntime

protocol DeviceArrayProtocol :
    DeviceAddressible, MutableDeviceCollection,
    RangeReplaceableCollection, ExpressibleByArrayLiteral
{
    associatedtype Buffer : AnyObject
    var capacity: Int { get }
    subscript(i: Int) -> Iterator.Element { get set }
    var buffer: Buffer { get set }
    init()
    init(device: Device)
    init(_ buffer: Buffer)
}

extension DeviceArrayProtocol where Buffer : DeviceArrayBufferProtocol {
    var capacity: Int {
        return buffer.capacity
    }
}

public struct DeviceArray<Element> : DeviceCollection, DeviceArrayProtocol {

    public typealias Index = Int
    public typealias IndexDistance = Int
    public typealias SubSequence = DeviceArray<Element>

    var buffer: DeviceArrayBuffer<Element>

    /// Copy on write
    private var mutatingBuffer: DeviceArrayBuffer<Element> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) {
                buffer = DeviceArrayBuffer(buffer)
            }
            return buffer
        }
    }

    public var device: Device {
        return buffer.device
    }

    init(_ buffer: DeviceArrayBuffer<Element>) {
        self.buffer = buffer
    }

    /// Creates an empty instance.
    public init() {
        buffer = DeviceArrayBuffer(device: Device.current, capacity: 0)
    }

    /// Creates an empty instance.
    public init(device: Device) {
        buffer = DeviceArrayBuffer(device: device, capacity: 0)
    }

    public init(device: Device, capacity: Int) {
        buffer = DeviceArrayBuffer(device: device, capacity: capacity)
    }

    public init(capacity: Int) {
        buffer = DeviceArrayBuffer(device: Device.current, capacity: capacity)
    }

    public init(repeating element: Element, count: Int) {
        buffer = DeviceArrayBuffer(repeating: element, count: count, device: Device.current)
    }

    public init(repeating element: Element, count: Int, device: Device) {
        buffer = DeviceArrayBuffer(repeating: element, count: count, device: device)
    }

    public init<C: Collection>(_ elements: C) where
        C.Iterator.Element == Element, C.IndexDistance == Int
    {
        buffer = DeviceArrayBuffer(elements, device: Device.current)
    }

    public init<C: Collection>(_ elements: C, device: Device) where
        C.Iterator.Element == Element, C.IndexDistance == Int
    {
        buffer = DeviceArrayBuffer(elements, device: device)
    }

    public init(arrayLiteral elements: Element...) {
        buffer = DeviceArrayBuffer(elements, device: Device.current)
    }

    public init(_ other: DeviceArray<Element>) {
        self = other
    }

    private init(viewing buffer: DeviceArrayBuffer<Element>, range: Range<Int>) {
        self.buffer = DeviceArrayBuffer(viewing: buffer, in: range)
    }

    public var hostArray: [Element] {
        var elements: [Element] = []
        elements.reserveCapacity(count)
        /// Temporary array copy solution
        var temp = UnsafeMutablePointer<Element>.allocate(capacity: count)
        temp.assign(fromDevice: buffer.startAddress, count: count)
        elements.append(contentsOf: UnsafeBufferPointer(start: temp, count: count))
        temp.deallocate(capacity: count)
        return elements
    }

    public var count: Int {
        return buffer.count
    }

    public func index(after i: Int) -> Int {
        return i + 1
    }

    public func index(before i: Int) -> Int {
        return i - 1
    }

    public var startIndex: Int {
        return 0
    }

    public var endIndex: Int {
        return buffer.count
    }

    public var indices: CountableRange<Int> {
        return 0..<count
    }
    /// Replaces the specified subrange of elements with the given collection.
    public mutating func replaceSubrange<C : Collection>
        (_ subrange: Range<Int>, with newElements: C) where C.Iterator.Element == DeviceValue<Element> {
        let subrange = CountableRange(subrange)
        for (index, element) in zip(subrange, newElements) {
            self[index] = element
        }
    }

    public subscript(i: Int) -> DeviceValue<Element> {
        get {
            return DeviceValue(buffer: DeviceValueBuffer(viewing: buffer, offsetBy: i))
        }
        set {
            var newValue = newValue
            mutatingBuffer[bufferIndex(fromLocal: i)] = newValue.mutatingBuffer
        }
    }

    @inline(__always)
    private func bufferRange(fromLocal localRange: Range<Int>) -> Range<Int> {
        return (buffer.startIndex + localRange.lowerBound)..<(buffer.startIndex + localRange.upperBound)
    }

    @inline(__always)
    private func bufferIndex(fromLocal localIndex: Int) -> Int {
        return localIndex + buffer.startIndex
    }

    public subscript(range: Range<Int>) -> DeviceArray<Element> {
        get {
            return DeviceArray(viewing: buffer, range: bufferRange(fromLocal: range))
        }
        mutating set {
            var newValue = newValue
            mutatingBuffer[bufferRange(fromLocal: range)] = newValue.mutatingBuffer
        }
    }

    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result {
        var startAddress = mutatingBuffer.startAddress
        return try body(&startAddress)
    }

    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try body(UnsafeDevicePointer(buffer.startAddress))
    }

}

public extension Array {

    public init(_ deviceElements: DeviceArray<Element>) {
        self = deviceElements.hostArray
    }

}
