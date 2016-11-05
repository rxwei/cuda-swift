//
//  Collection.swift
//  Warp
//
//  Created by Richard Wei on 11/1/16.
//
//

import CUDARuntime

public protocol DeviceAddressible {
    associatedtype Element
    func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result
    mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result
}

public protocol DeviceCollection : DeviceAddressible, RandomAccessCollection {
    typealias Index = Int
    typealias IndexDistance = Int
    associatedtype Element
    associatedtype SubSequence : RandomAccessCollection
    var hostArray: [Element] { get }
    subscript(index: Int) -> Iterator.Element { get }
}

public protocol MutableDeviceCollection : DeviceCollection {
    subscript(index: Int) -> Iterator.Element { get set }
}
