//
//  Array.swift
//  CUDA
//
//  Created by Richard Wei on 10/19/16.
//
//

import CCUDARuntime

public struct DeviceArray<Element> : RandomAccessCollection, RangeReplaceableCollection, ExpressibleByArrayLiteral {
    public typealias Index = Int
    public typealias IndexDistance = Int
    public typealias SubSequence = DeviceArray<Element>

    private var buffer: DeviceArrayBuffer<Element>

    /// Copy on write
    private var mutatingBuffer: DeviceArrayBuffer<Element> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) {
                buffer = DeviceArrayBuffer(buffer)
            }
            return buffer
        }
    }

    /// Creates an empty instance.
    public init() {
        buffer = DeviceArrayBuffer(capacity: 0)
    }

    public init(capacity: Int) {
        buffer = DeviceArrayBuffer(capacity: capacity)
    }

    public init<C: Collection>(fromHost elements: C) where
        C.Iterator.Element == Element, C.IndexDistance == Int
    {
        buffer = DeviceArrayBuffer(capacity: elements.count)
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
        temp.assign(fromDevice: buffer.baseAddress, count: count)
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
        return count
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
            return DeviceValue(buffer: ManagedDeviceBuffer(viewing: buffer, offset: i))
        }
        set {
            newValue.withUnsafeDevicePointer { newValuePtr in
                mutatingBuffer.baseAddress.advanced(by: i).assign(from: newValuePtr)
            }
        }
    }

    private init(viewing buffer: DeviceArrayBuffer<Element>, range: Range<Int>) {
        self.buffer = DeviceArrayBuffer(viewing: buffer, range: range)
    }

    public subscript(range: Range<Int>) -> DeviceArray<Element> {
        get {
            return DeviceArray(viewing: buffer, range: range)
        }
        mutating set {
            mutatingBuffer.baseAddress.assign(
                from: newValue.buffer.baseAddress,
                count: Swift.min(range.count, newValue.count)
            ///        ^ adding qualifier to disambiguate from `min(by:)`
            /// Compiler bug: SR-3051
            )
        }
    }

    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try body(mutatingBuffer.baseAddress)
    }

    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try body(UnsafeDevicePointer(buffer.baseAddress))
    }

}

public extension Array {

    public init(_ elementsOnDevice: DeviceArray<Element>) {
        self = elementsOnDevice.copyToHost()
    }
    
}
