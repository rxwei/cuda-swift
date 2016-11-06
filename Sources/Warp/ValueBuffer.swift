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
    var device: Device { get }
    var owner: AnyObject? { get }
    var baseAddress: UnsafeMutableDevicePointer<Element> { get }
    init()
    init(device: Device)
    init(viewing other: Self)
}

protocol DeviceArrayViewingBufferProtocol : DeviceBufferProtocol {
    init<ArrayBuffer: DeviceArrayBufferProtocol>(viewing other: ArrayBuffer, offsetBy offset: Int)
        where ArrayBuffer.Element == Element
}

final class DeviceValueBuffer<Element> : DeviceArrayViewingBufferProtocol {
    let device: Device
    let baseAddress: UnsafeMutableDevicePointer<Element>
    let owner: AnyObject?
    private var retainee: Element?

    init() {
        device = Device.current
        baseAddress = UnsafeMutableDevicePointer<Element>.allocate(capacity: 1)
        owner = nil
    }

    convenience init(device: Device) {
        let contextualDevice = Device.current
        if device == contextualDevice {
            self.init()
        } else {
            Device.current = device
            self.init()
            Device.current = contextualDevice
        }
    }

    convenience init(_ other: DeviceValueBuffer<Element>) {
        self.init(device: other.device)
        baseAddress.assign(from: other.baseAddress)
        retainee = other.retainee
    }

    init(viewing other: DeviceValueBuffer<Element>) {
        device = other.device
        baseAddress = other.baseAddress
        owner = other
    }

    init<Buffer: DeviceArrayBufferProtocol>
        (viewing arrayBuffer: Buffer, offsetBy offset: Int) where Buffer.Element == Element
    {
        device = arrayBuffer.device
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
