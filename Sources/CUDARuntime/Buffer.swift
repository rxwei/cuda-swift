//
//  Buffer.swift
//  CUDA
//
//  Created by Richard Wei on 10/26/16.
//
//

public protocol DeviceAddressible {
    associatedtype Element
    var unsafePointer: UnsafeDevicePointer<Element> { get }
    func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result
    mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result
}

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
    var retainee: Any? { get set }

    init(capacity: Int)
    init(viewing other: Self, in range: Range<Int>)

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

    required init() {
        baseAddress = UnsafeMutableDevicePointer<Element>.allocate(capacity: 1)
        owner = nil
    }

    convenience init(_ other: DeviceValueBuffer<Element>) {
        self.init()
        baseAddress.assign(from: other.baseAddress)
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
    var retainee: Any?

    init(capacity: Int) {
        self.capacity = capacity
        baseAddress = UnsafeMutableDevicePointer<Element>.allocate(capacity: capacity)
        startIndex = 0
        endIndex = capacity
        owner = nil
    }

    init(viewing other: DeviceArrayBuffer<Element>) {
        capacity = other.capacity
        baseAddress = other.baseAddress
        startIndex = other.startIndex
        endIndex = other.endIndex
        owner = other
    }

    /// Copy elements from the other buffer
    convenience init(_ other: DeviceArrayBuffer<Element>) {
        self.init(capacity: other.count)
        baseAddress.assign(from: other.baseAddress.advanced(by: other.startIndex), count: count)
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
            baseAddress.advanced(by: bounds.lowerBound)
                       .assign(from: newValue.baseAddress.advanced(by: newValue.startIndex),
                               count: Swift.min(bounds.count, count))
        }
    }

    subscript(i: Int) -> DeviceValueBuffer<Element> {
        get {
            return DeviceValueBuffer(viewing: self, offsetBy: i)
        }
        set {
            baseAddress.advanced(by: i).assign(from: newValue.baseAddress)
        }
    }

}
