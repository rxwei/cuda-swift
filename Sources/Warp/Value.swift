//
//  Value.swift
//  Warp
//
//  Created by Richard Wei on 10/23/16.
//
//

import CUDARuntime

/// Value view on CUDA device
public struct DeviceValue<Element> : MutableDeviceAddressable {

    var buffer: DeviceValueBuffer<Element>

    /// Copy on write
    var mutatingBuffer: DeviceValueBuffer<Element> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) || buffer.owner != nil {
                buffer = DeviceValueBuffer(buffer)
            }
            return buffer
        }
    }

    init(buffer: DeviceValueBuffer<Element>) {
        self.buffer = buffer
    }

    /// Device on which the value is allocated
    public var device: Device {
        return buffer.device
    }

    /// Dereferenced value, copied from device
    public var value: Element {
        get {
            return buffer.value
        }
        set {
            mutatingBuffer.value = newValue
        }
    }

    /// Initialize a value on device memory
    ///
    /// - Parameter initialValue: initial value, or arbitrary unsafe value if
    /// not specified
    public init(_ initialValue: Element? = nil) {
        buffer = DeviceValueBuffer()
        if let initialValue = initialValue {
            buffer.value = initialValue
        }
    }

    /// Initialize a value on device memory
    ///
    /// - Parameters:
    ///   - initialValue: initial value, or arbitrary unsafe
    ///   - device: device to allocate memory on
    /// value if not specified
    public init(_ initialValue: Element? = nil, device: Device) {
        buffer = DeviceValueBuffer(device: device)
        if let initialValue = initialValue {
            buffer.value = initialValue
        }
    }

    /// Calls a closure with a mutable pointer to the value.
    ///
    /// - Parameter body: body closure with a pointer made available
    /// - Returns: whatever the closure returns
    /// - Throws: whatever the closure throws
    /// - Note: Do not return the pointer from the closure
    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result {
        var baseAddress = mutatingBuffer.baseAddress
        return try body(&baseAddress)
    }

    /// Calls a closure with a pointer to the value.
    ///
    /// - Parameter body: body closure with a pointer made available
    /// - Returns: whatever the closure returns
    /// - Throws: whatever the closure throws
    /// - Note: Do not return the pointer from the closure
    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try body(UnsafeDevicePointer(buffer.baseAddress))
    }

    /// Get the mutable device pointer to the value.
    /// - Note: This should **only** be used when the lifetime of the pointer is
    /// shorter than the lifetime of the value. If you are not sure about the life
    /// time of the value, use `.withUnsafeMutableDevicePointer(...)` instead!
    public var unsafeMutableDevicePointer: UnsafeMutableDevicePointer<Element> {
        mutating get {
            return mutatingBuffer.baseAddress
        }
    }

    /// Get the device pointer to the value.
    /// - Note: This should **only** be used when the lifetime of the pointer is
    /// shorter than the lifetime of the value. If you are not sure about the life
    /// time of the value, use `.withUnsafeMutableDevicePointer(...)` instead!
    public var unsafeDevicePointer: UnsafeDevicePointer<Element> {
        return UnsafeDevicePointer(buffer.baseAddress)
    }

}

// MARK: - Underlying mutable device collection
extension DeviceValue where Element : MutableDeviceCollection {

    /// Get or set the i-th element of the underlying array when the value 
    /// is an array.
    ///
    /// - Parameter i: index
    public subscript(i: Int) -> Element.Iterator.Element {
        set {
            value[i] = newValue
        }
        get {
            return value[i]
        }
    }

}

// MARK: - Underlying device collection
extension DeviceValue where Element : DeviceCollection {

    /// Get an array on host memory by copying from CUDA device
    public var hostArray: [Element.Element] {
        return value.hostArray
    }
    
}

// MARK: - Underlying device collection whose element type is also device collection
extension DeviceValue where Element : DeviceCollection, Element.Element : DeviceCollection {

    /// Get an array on host memory by copying from CUDA device
    public var hostArray: [[Element.Element.Element]] {
        return value.hostArray.map{$0.hostArray}
    }
    
}
