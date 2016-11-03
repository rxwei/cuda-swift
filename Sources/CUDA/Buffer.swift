//
//  Buffer.swift
//  Warp
//
//  Created by Richard Wei on 10/26/16.
//
//

import CUDARuntime

protocol DeviceBufferProtocol : class {
    associatedtype Element
    var owner: AnyObject? { get }
    var baseAddress: UnsafeMutableDevicePointer<Element> { get }
    init()
    init(viewing other: Self)
}

protocol ArrayViewingBufferProtocol : DeviceBufferProtocol {
    init<ArrayBuffer: DeviceArrayBufferProtocol>(viewing other: ArrayBuffer, offsetBy offset: Int)
        where ArrayBuffer.Element == Element
}

protocol DeviceArrayBufferProtocol : DeviceBufferProtocol, MutableCollection, RandomAccessCollection {
    typealias Index = Int
    typealias Indices = CountableRange<Int>

    var baseAddress: UnsafeMutableDevicePointer<Element> { get }
    var capacity: Int { get }

    init(capacity: Int)
    init(viewing other: Self, in range: Range<Int>)

    var startAddress: UnsafeMutableDevicePointer<Element> { get }
    var endAddress: UnsafeMutableDevicePointer<Element> { get }

    subscript(i: Int) -> DeviceValueBuffer<Element> { get set }
}

extension DeviceArrayBufferProtocol {
    init() {
        self.init(capacity: 0)
    }
}

final class DeviceValueBuffer<Element> : ArrayViewingBufferProtocol {
    let baseAddress: UnsafeMutableDevicePointer<Element>
    let owner: AnyObject?
    private var retainee: Element?

    required init() {
        baseAddress = UnsafeMutableDevicePointer<Element>.allocate(capacity: 1)
        owner = nil
    }

    convenience init(_ other: DeviceValueBuffer<Element>) {
        self.init()
        baseAddress.assign(from: other.baseAddress)
        retainee = other.retainee
    }

    required init(viewing other: DeviceValueBuffer<Element>) {
        baseAddress = other.baseAddress
        owner = other
    }

    required init<Buffer: DeviceArrayBufferProtocol>
        (viewing arrayBuffer: Buffer, offsetBy offset: Int) where Buffer.Element == Element
    {
        baseAddress = arrayBuffer.baseAddress.advanced(by: arrayBuffer.startIndex + offset)
        owner = arrayBuffer
    }

    var value: Element {
        get {
            return baseAddress.load()
        }
        set {
            retainee = newValue
            baseAddress.assign(newValue)
        }
    }

    deinit {
        if owner == nil {
            baseAddress.deallocate()
        }
    }

}

final class DeviceArrayBuffer<Element> : DeviceArrayBufferProtocol {
    typealias SubSequence = DeviceArrayBuffer<Element>

    let baseAddress: UnsafeMutableDevicePointer<Element>
    let capacity: Int
    let startIndex: Int, endIndex: Int
    var owner: AnyObject?
    private var retainees: [Element]?
    private var valueRetainees: [DeviceValueBuffer<Element>?]

    init(capacity: Int) {
        self.capacity = capacity
        baseAddress = UnsafeMutableDevicePointer<Element>.allocate(capacity: capacity)
        startIndex = 0
        endIndex = capacity
        owner = nil
        valueRetainees = Array(repeating: nil, count: capacity)
    }

    init(viewing other: DeviceArrayBuffer<Element>) {
        capacity = other.capacity
        baseAddress = other.baseAddress
        startIndex = other.startIndex
        endIndex = other.endIndex
        owner = other
        retainees = other.retainees
        valueRetainees = other.valueRetainees
    }

    init(viewing other: DeviceArrayBuffer<Element>, in range: Range<Int>) {
        baseAddress = other.baseAddress
        guard other.startIndex <= range.lowerBound &&
            other.endIndex >= range.upperBound else {
            fatalError("Array index out of bounds")
        }
        capacity = other.capacity
        startIndex = range.lowerBound
        endIndex = range.upperBound
        owner = other
        retainees = other.retainees
        valueRetainees = other.valueRetainees
    }

    convenience init<C: Collection>(_ elements: C) where
        C.Iterator.Element == Element, C.IndexDistance == Int
    {
        self.init(capacity: elements.count)
        var elements = Array(elements)
        baseAddress.assign(fromHost: &elements, count: elements.count)
        retainees = elements
    }

    convenience init(repeating repeatedValue: Element, count: Int) {
        self.init(Array(repeating: repeatedValue, count: count))
    }

    convenience init(_ other: DeviceArrayBuffer<Element>) {
        self.init(capacity: other.count)
        retainees = other.retainees
        baseAddress.assign(from: other.startAddress, count: other.count)
    }
    
    deinit {
        if owner == nil {
            baseAddress.deallocate()
        }
    }

    var count: Int {
        return endIndex - startIndex
    }

    func index(after i: Int) -> Int {
        return i + 1
    }

    func index(before i: Int) -> Int {
        return i - 1
    }

    var indices: CountableRange<Int> {
        return startIndex..<endIndex
    }

    var startAddress: UnsafeMutableDevicePointer<Element> {
        return baseAddress.advanced(by: startIndex)
    }

    var endAddress: UnsafeMutableDevicePointer<Element> {
        return baseAddress.advanced(by: endIndex)
    }

    /// Accesses the subsequence bounded by the given range.
    ///
    /// - Parameter bounds: A range of the collection's indices. The upper and
    ///   lower bounds of the `bounds` range must be valid indices of the
    ///   collection.
    subscript(bounds: Range<Int>) -> DeviceArrayBuffer<Element> {
        get {
            return DeviceArrayBuffer(viewing: self, in: bounds)
        }
        set {
            guard bounds.lowerBound >= startIndex && bounds.upperBound <= endIndex else {
                fatalError("Array index out of bounds")
            }
            for (i, valueBuf) in zip(CountableRange(bounds), newValue) {
                valueRetainees[i] = valueBuf
            }
            baseAddress.advanced(by: bounds.lowerBound)
                       .assign(from: newValue.startAddress,
                               count: Swift.min(bounds.count, count))
        }
    }

    subscript(i: Int) -> DeviceValueBuffer<Element> {
        get {
            return DeviceValueBuffer(viewing: self, offsetBy: i)
        }
        set {
            valueRetainees[i] = newValue
            baseAddress.advanced(by: i).assign(from: newValue.baseAddress)
        }
    }

}
