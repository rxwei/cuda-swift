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

    /// Raw base address on CUDA device
    let deviceAddress: UnsafeMutableRawPointer

    /// Initializer for other CUDA Runtime API calls within the module
    public init(_ deviceAddress: UnsafeMutableRawPointer) {
        self.deviceAddress = deviceAddress
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
            deviceAddress.advanced(by: n * MemoryLayout<Pointee>.stride)
        )
    }

    public func distance(to other: UnsafeMutableDevicePointer<Pointee>) -> Int {
        return self.deviceAddress.distance(to: other.deviceAddress)
    }

    public func predecessor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(deviceAddress - MemoryLayout<Pointee>.stride)
    }

    public func successor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(deviceAddress + MemoryLayout<Pointee>.stride)
    }

    /// Pointee **copied** from device
    /// Avoid doing this often because it's expensive
    /// - complexity: O(size)
    public var pointee: Pointee {
        var pointee: Pointee?
        !!cudaMemcpy(&pointee, deviceAddress,
                     MemoryLayout<Pointee>.stride, cudaMemcpyDeviceToHost)
        return pointee!
    }

    public func initialize(to value: Pointee, count: Int = 1) {
        var value = value
        !!cudaMemcpy(deviceAddress, &value,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
    }

    /// TODO: Change argument type to UnsafeDevicePointer once it's implemented
//    public func initialize(mutating pointer: UnsafePointer<Pointee>, count: Int) {
//        !!cudaMemcpy(self.pointer, pointer,
//                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
//    }

    public func initialize<C: Collection>(from elements: C)
        where C.Iterator.Element == Pointee, C.IndexDistance == Int {
        ContiguousArray(elements).withUnsafeBufferPointer { ptr in
            !!cudaMemcpy(self.deviceAddress, ptr.baseAddress,
                         elements.count * MemoryLayout<Pointee>.stride,
                         cudaMemcpyHostToDevice)
        }
    }

    public func assign(from pointer: UnsafePointer<Pointee>, count: Int) {
        !!cudaMemcpy(self.deviceAddress, pointer,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
    }

    public func withRawAddress<Result>
        (body: (UnsafeRawPointer) throws -> Result) rethrows -> Result {
        return try body(deviceAddress)
    }

    public func withMutableRawAddress<Result>
        (body: (UnsafeMutableRawPointer) throws -> Result) rethrows -> Result {
        return try body(deviceAddress)
    }

    public subscript(i: Int) -> Pointee {
        get {
            return self.advanced(by: i).pointee
        }
        nonmutating set {
            var newValue = newValue
            !!cudaMemcpy(advanced(by: i).deviceAddress, &newValue,
                         MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
        }
    }
    
}
