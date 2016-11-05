//
//  Buffer.swift
//  Warp
//
//  Created by Richard Wei on 10/26/16.
//
//

import CUDARuntime

protocol DeviceArrayBufferProtocol : DeviceBufferProtocol, MutableCollection, RandomAccessCollection {
    typealias Index = Int
    typealias Indices = CountableRange<Int>

    var baseAddress: UnsafeMutableDevicePointer<Element> { get }
    var capacity: Int { get }

    init(device: Device, capacity: Int)
    init(viewing other: Self, in range: Range<Int>)

    var startAddress: UnsafeMutableDevicePointer<Element> { get }
    var endAddress: UnsafeMutableDevicePointer<Element> { get }

    subscript(i: Int) -> DeviceValueBuffer<Element> { get set }
}

extension DeviceArrayBufferProtocol {
    var startAddress: UnsafeMutableDevicePointer<Element> {
        return baseAddress.advanced(by: startIndex)
    }

    var endAddress: UnsafeMutableDevicePointer<Element> {
        return baseAddress.advanced(by: endIndex)
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
}

final class DeviceArrayBuffer<Element> : DeviceArrayBufferProtocol {
    typealias SubSequence = DeviceArrayBuffer<Element>

    let device: Device
    let baseAddress: UnsafeMutableDevicePointer<Element>
    let capacity: Int
    let startIndex: Int, endIndex: Int
    var owner: AnyObject?
    private var retainees: [Element]?
    private var valueRetainees: [DeviceValueBuffer<Element>?]

    convenience init(device: Device) {
        self.init(device: device, capacity: 0)
    }

    init(device: Device, capacity: Int) {
        self.capacity = capacity
        self.device = device
        baseAddress = UnsafeMutableDevicePointer<Element>.allocate(capacity: capacity)
        startIndex = 0
        endIndex = capacity
        owner = nil
        valueRetainees = Array(repeating: nil, count: capacity)
    }

    init(viewing other: DeviceArrayBuffer<Element>) {
        capacity = other.capacity
        device = other.device
        baseAddress = other.baseAddress
        startIndex = other.startIndex
        endIndex = other.endIndex
        owner = other
        retainees = other.retainees
        valueRetainees = other.valueRetainees
    }

    init(viewing other: DeviceArrayBuffer<Element>, in range: Range<Int>) {
        device = other.device
        capacity = other.capacity
        baseAddress = other.baseAddress
        guard other.startIndex <= range.lowerBound &&
            other.endIndex >= range.upperBound else {
            fatalError("Array index out of bounds")
        }
        startIndex = range.lowerBound
        endIndex = range.upperBound
        owner = other
        retainees = other.retainees
        valueRetainees = other.valueRetainees
    }

    convenience init<C: Collection>(_ elements: C, device: Device) where
        C.Iterator.Element == Element, C.IndexDistance == Int
    {
        self.init(device: device, capacity: elements.count)
        var elements = Array(elements)
        baseAddress.assign(fromHost: &elements, count: elements.count)
        retainees = elements
    }

    convenience init(repeating repeatedValue: Element, count: Int, device: Device) {
        self.init(Array(repeating: repeatedValue, count: count), device: device)
    }

    convenience init(_ other: DeviceArrayBuffer<Element>) {
        self.init(device: other.device, capacity: other.count)
        retainees = other.retainees
        baseAddress.assign(from: other.startAddress, count: other.count)
    }
    
    deinit {
        if owner == nil {
            baseAddress.deallocate()
        }
    }

    var indices: CountableRange<Int> {
        return startIndex..<endIndex
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
