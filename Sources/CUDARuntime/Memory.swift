//
//  Memory.swift
//  CUDA
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCUDARuntime

/// Richard's thoughts on UnsafeMutableDevicePointer:
/// - Since it's explicitly unsafe, we can avoid throwing CUDA errors here
///   but instead unsafely `try!` everything, for the ease of use
/// - For this, we created the `!!` operator!

/// Mutable Device Pointer using Runtime API
public struct UnsafeMutableDevicePointer<Pointee> : Equatable, Hashable, Strideable {

    public typealias Stride = Int

    /// Raw address on CUDA device
    public let deviceAddress: UnsafeMutablePointer<Pointee>

    /// Convert from raw memory address on graphic device
    /// - parameter deviceAddress: address on graphic device
    /// - note: Initializer for other CUDA Runtime API calls within the module
    /// Maybe we need a safer solution, e.g. only treat 
    /// `init(_: UnsafeMutablePointer<Float>)` as value preserving type conversion,
    /// while using an argument-labeled initializer for conversion from
    /// UnsafeMutableRawPointer, e.g.
    /// `init(assumingMemoryBoundFrom: UnsafeMutableRawPointer)`
    /// For now, let's leave it as is
    public init(_ deviceAddress: UnsafeMutableRawPointer) {
        self.deviceAddress = deviceAddress.assumingMemoryBound(to: Pointee.self)
    }

    public static func allocate(capacity: Int) -> UnsafeMutableDevicePointer<Pointee> {
        var pointer: UnsafeMutableRawPointer?
        !!cudaMalloc(&pointer, capacity * MemoryLayout<Pointee>.stride)
        return UnsafeMutableDevicePointer(pointer!)
    }

    public func deallocate() {
        !!cudaFree(deviceAddress)
    }

    public var hashValue: Int {
        return deviceAddress.hashValue
    }

    public static func ==(lhs: UnsafeMutableDevicePointer<Pointee>,
                          rhs: UnsafeMutableDevicePointer<Pointee>) -> Bool {
        return lhs.deviceAddress == rhs.deviceAddress
    }

    public func advanced(by n: Int) -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(
            deviceAddress.advanced(by: n)
        )
    }

    public func distance(to other: UnsafeMutableDevicePointer<Pointee>) -> Int {
        return self.deviceAddress.distance(to: other.deviceAddress)
    }

    public func predecessor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(deviceAddress - 1)
    }

    public func successor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(deviceAddress + 1)
    }

    /// Pointee **copied** from device
    /// Avoid doing this often because it's expensive
    /// - complexity: O(size)
    public func load() -> Pointee {
        var pointee: Pointee?
        !!cudaMemcpy(&pointee, deviceAddress,
                     MemoryLayout<Pointee>.size, cudaMemcpyDeviceToHost)
        return pointee!
    }

    public func assign(_ value: Pointee) {
        var value = value
        !!cudaMemcpy(deviceAddress, &value,
                     MemoryLayout<Pointee>.size, cudaMemcpyHostToDevice)
    }

    public func assign(_ value: Pointee, count: Int) {
        var value = value
        !!cudaMemcpy(deviceAddress, &value,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
    }

    public func assign<C: Collection>(fromHost elements: C)
        where C.Iterator.Element == Pointee, C.IndexDistance == Int {
        ContiguousArray(elements).withUnsafeBufferPointer { ptr in
            !!cudaMemcpy(self.deviceAddress, ptr.baseAddress,
                         elements.count * MemoryLayout<Pointee>.stride,
                         cudaMemcpyHostToDevice)
        }
    }

    public func assign(fromHost pointer: UnsafePointer<Pointee>) {
        !!cudaMemcpy(self.deviceAddress, pointer,
                     MemoryLayout<Pointee>.size, cudaMemcpyHostToDevice)
    }

    public func assign(fromHost pointer: UnsafePointer<Pointee>, count: Int) {
        !!cudaMemcpy(self.deviceAddress, pointer,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
    }

    public func assign(from pointer: UnsafeMutableDevicePointer<Pointee>) {
        !!cudaMemcpy(self.deviceAddress, pointer.deviceAddress,
                     MemoryLayout<Pointee>.size, cudaMemcpyDeviceToDevice)
    }

    public func assign(from pointer: UnsafeMutableDevicePointer<Pointee>, count: Int) {
        !!cudaMemcpy(self.deviceAddress, pointer.deviceAddress,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyDeviceToDevice)
    }

    public func copyBytes(toHost pointer: UnsafeMutablePointer<Pointee>, count: Int) {
        !!cudaMemcpy(pointer, deviceAddress,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyDeviceToHost)
    }

    public subscript(i: Int) -> Pointee {
        get {
            return self.advanced(by: i).load()
        }
        nonmutating set {
            var newValue = newValue
            !!cudaMemcpy(advanced(by: i).deviceAddress, &newValue,
                         MemoryLayout<Pointee>.size, cudaMemcpyHostToDevice)
        }
    }
    
}

public extension UnsafeMutableDevicePointer {

    public func withDeviceAddress<Result>
        (body: (UnsafePointer<Pointee>) throws -> Result) rethrows -> Result {
        return try body(deviceAddress)
    }

    public func withMemoryRebound<T, Result>
        (to type: T.Type, capacity: Int,
         _ body: (UnsafeMutableDevicePointer<T>) throws -> Result) rethrows -> Result {
        return try deviceAddress.withMemoryRebound(to: T.self, capacity: capacity) { ptr -> Result in
            try body(UnsafeMutableDevicePointer<T>(ptr))
        }
    }

}

public extension UnsafeMutablePointer {

    public mutating func assign(fromDevice devPtr: UnsafeMutableDevicePointer<Pointee>, count: Int) {
        devPtr.copyBytes(toHost: self, count: count)
    }
    
}
