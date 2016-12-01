//
//  Collection.swift
//  Warp
//
//  Created by Richard Wei on 11/1/16.
//
//

import CUDARuntime
@_exported import struct CUDARuntime.Device

public protocol DeviceAddressable {
    associatedtype Element
    var device: Device { get }
    func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result
    mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result
    var unsafeMutableDevicePointer: UnsafeMutableDevicePointer<Element> { mutating get }
    var unsafeDevicePointer: UnsafeDevicePointer<Element> { get }
}

public protocol DeviceCollection : DeviceAddressable, RandomAccessCollection {
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
