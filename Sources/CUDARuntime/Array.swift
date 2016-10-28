//
//  Array.swift
//  CUDA
//
//  Created by Richard Wei on 10/19/16.
//
//

import CCUDARuntime

public protocol DeviceCollection : RandomAccessCollection {
    typealias Index = Int
    typealias IndexDistance = Int
    associatedtype Element
    associatedtype SubSequence : RandomAccessCollection
    func copyToHost() -> [Element]
    subscript(index: Int) -> Iterator.Element { get }
}

public protocol MutableDeviceCollection : DeviceCollection {
    subscript(index: Int) -> Iterator.Element { get set }
}

protocol DeviceArrayProtocol :
    DeviceAddressible, MutableDeviceCollection,
    RangeReplaceableCollection, ExpressibleByArrayLiteral
{
    associatedtype Buffer : AnyObject
    var capacity: Int { get }
    subscript(i: Int) -> Iterator.Element { get set }
    var buffer: Buffer { get set }
    init(_ buffer: Buffer)
}

extension DeviceArrayProtocol where Buffer : DeviceArrayBufferProtocol {
    var capacity: Int {
        return buffer.capacity
    }
}

public struct DeviceArray<Element> : RandomAccessCollection, DeviceArrayProtocol
{
    public typealias Index = Int
    public typealias IndexDistance = Int
    public typealias SubSequence = DeviceArray<Element>

    var buffer: DeviceArrayBuffer<Element>
    private var retainedReference: Any?

    /// Copy on write
    private var mutatingBuffer: DeviceArrayBuffer<Element> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) {
                buffer = DeviceArrayBuffer(buffer)
            }
            return buffer
        }
    }

    public var unsafePointer: UnsafeDevicePointer<Element> {
        return UnsafeDevicePointer(buffer.baseAddress.advanced(by: buffer.startIndex))
    }

    init(_ buffer: DeviceArrayBuffer<Element>) {
        self.buffer = buffer
    }

    /// Creates an empty instance.
    public init() {
        buffer = DeviceArrayBuffer(capacity: 0)
    }

    public init(capacity: Int) {
        buffer = DeviceArrayBuffer(capacity: capacity)
    }

    public init(repeating element: Element, count: Int) {
        buffer = DeviceArrayBuffer(capacity: count)
        buffer.baseAddress.assign(element, count: count)
    }

    public init<C: Collection>(fromHost elements: C) where
        C.Iterator.Element == Element, C.IndexDistance == Int
    {
        buffer = DeviceArrayBuffer(capacity: elements.count)
        buffer.retainee = elements
        buffer.baseAddress.assign(fromHost: elements)
    }

    public init(arrayLiteral elements: Element...) {
        self.init(fromHost: elements)
    }

    public init(_ other: DeviceArray<Element>) {
        self = other
    }

    public func copyToHost() -> [Element] {
        var elements: [Element] = []
        elements.reserveCapacity(count)
        /// Temporary array copy solution
        var temp = UnsafeMutablePointer<Element>.allocate(capacity: count)
        temp.assign(fromDevice: buffer.baseAddress.advanced(by: buffer.startIndex), count: count)
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
            return DeviceValue(buffer:
                DeviceValueBuffer(viewing: buffer, offsetBy: i)
            )
        }
        set {
            mutatingBuffer[bufferIndex(fromLocal: i)] = newValue.buffer
        }
    }

    private init(viewing buffer: DeviceArrayBuffer<Element>, range: Range<Int>) {
        self.buffer = DeviceArrayBuffer(viewing: buffer, in: range)
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
            mutatingBuffer[bufferRange(fromLocal: range)] = newValue.buffer
        }
    }

    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result {
        let buffer = mutatingBuffer
        var baseAddress = buffer.baseAddress.advanced(by: buffer.startIndex)
        return try body(&baseAddress)
    }

    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try body(UnsafeDevicePointer(buffer.baseAddress.advanced(by: buffer.startIndex)))
    }

}

public extension Array {

    public init(_ elementsOnDevice: DeviceArray<Element>) {
        self = elementsOnDevice.copyToHost()
    }

}
