//
//  Array.swift
//  CUDA
//
//  Created by Richard Wei on 10/19/16.
//
//

import CCUDARuntime

/// Maybe we need a CoW value type DeviceArray
/// Goal:
/// - hierachy:
///   - public struct DeviceArray<Element>
///   - private class DeviceArrayBuffer<Element>
///   - public struct UnsafeMutableDeviceBufferPointer<Element>
///   - public struct UnsafeMutableDevicePointer<Element>

public struct UnsafeMutableDeviceBufferPointer<Element> {

    public let baseAddress: UnsafeMutableDevicePointer<Element>?

    public let count: Int

    public init(start: UnsafeMutableDevicePointer<Element>?, count: Int) {
        self.baseAddress = start
        self.count = start == nil ? 0 : count
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

    public subscript(i: Int) -> Element {
        get {
            return baseAddress![i]
        }
        nonmutating set {
            baseAddress![i] = newValue
        }
    }

}
