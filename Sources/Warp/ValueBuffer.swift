//
//  ValueBuffer.swift
//  CUDA
//
//  Created by Richard Wei on 11/4/16.
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
