//
//  Buffer.swift
//  CUDA
//
//  Created by Richard Wei on 10/26/16.
//
//

import Foundation

open class ManagedDeviceBuffer<Element> {

    public let baseAddress: UnsafeMutableDevicePointer<Element>

    public let capacity: Int

    /// Optional strong reference to other buffer. Used when array produces a
    /// slice. It needs to hold on to the original buffer so that the original
    /// array makes a copy when it mutates.
    public let retainedReference: ManagedDeviceBuffer<Element>?

    public convenience init(_ other: ManagedDeviceBuffer<Element>) {
        self.init(capacity: other.capacity)
        baseAddress.assign(from: other.baseAddress)
    }

    public init(capacity: Int) {
        self.capacity = capacity
        baseAddress = UnsafeMutableDevicePointer<Element>.allocate(capacity: capacity)
        retainedReference = nil
    }

    public init(viewing other: ManagedDeviceBuffer<Element>) {
        capacity = other.capacity
        baseAddress = other.baseAddress
        retainedReference = other
    }

    public init(viewing other: ManagedDeviceBuffer<Element>, offset: Int) {
        capacity = other.capacity
        baseAddress = other.baseAddress + offset
        retainedReference = other
    }

    deinit {
        /// Deallocate if and only if `self` is owning the memory
        if retainedReference == nil {
            baseAddress.deallocate()
        }
    }

}

final class DeviceArrayBuffer<Element> : ManagedDeviceBuffer<Element> {

    let count: Int

    /// Copy elements from the other buffer
    public init(_ other: DeviceArrayBuffer<Element>) {
        count = other.count
        super.init(capacity: other.count)
        baseAddress.assign(from: other.baseAddress, count: count)
    }

    override init(capacity: Int) {
        count = capacity
        super.init(capacity: capacity)
    }

    init(viewing other: DeviceArrayBuffer<Element>) {
        count = other.capacity
        super.init(viewing: other)
    }

    init(viewing other: DeviceArrayBuffer<Element>, range: Range<Int>) {
        count = range.count
        super.init(viewing: other, offset: range.lowerBound)
    }

}
