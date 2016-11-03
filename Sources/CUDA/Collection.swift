//
//  Collection.swift
//  Warp
//
//  Created by Richard Wei on 11/1/16.
//
//

import Foundation

public protocol DeviceCollection : DeviceAddressible, RandomAccessCollection {
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
