//
//  Memory.swift
//  CUDA
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCUDARuntime

/// Richard's thoughts on UnsafeMutableDevicePointer:
/// - Since it's explicitly unsafe, we can avoid throwing errors here
///   but instead unsafely `try!` everything, for the ease of use
/// - For this, we created the `!!` operator!

/// Mutable Device Pointer using Runtime API
public struct UnsafeMutableDevicePointer<Pointee> : Equatable, Hashable, Strideable {

    public typealias Stride = Int

    public var baseAddress: UnsafeMutablePointer<Pointee>

    /// For other CUDA Runtime API calls within the module
    internal init(baseAddress: UnsafeMutableRawPointer) {
        self.baseAddress = baseAddress.assumingMemoryBound(to: Pointee.self)
    }

    public static func allocate(capacity: Int) -> UnsafeMutableDevicePointer<Pointee> {
        var pointer: UnsafeMutableRawPointer?
        !!cudaMalloc(&pointer, capacity * MemoryLayout<Pointee>.stride)
        return UnsafeMutableDevicePointer(baseAddress: pointer!)
    }

    public func deallocate() {
        !!cudaFree(baseAddress)
    }

    public var hashValue: Int {
        return baseAddress.hashValue
    }

    public static func ==(lhs: UnsafeMutableDevicePointer<Pointee>,
                          rhs: UnsafeMutableDevicePointer<Pointee>) -> Bool {
        return lhs.baseAddress == rhs.baseAddress
    }

    public func advanced(by n: Int) -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(baseAddress: baseAddress.advanced(by: n))
    }

    public func distance(to other: UnsafeMutableDevicePointer<Pointee>) -> Int {
        return self.baseAddress.distance(to: other.baseAddress)
    }

    public func predecessor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(baseAddress: baseAddress.predecessor())
    }

    public func successor() -> UnsafeMutableDevicePointer {
        return UnsafeMutableDevicePointer(baseAddress: baseAddress.successor())
    }

    public var pointee: Pointee {
        var pointee: Pointee?
        !!cudaMemcpy(&pointee, baseAddress, MemoryLayout<Pointee>.stride, cudaMemcpyDeviceToHost)
        return pointee!
    }

    public func move() -> Pointee {
        defer { !!cudaFree(baseAddress) }
        return pointee
    }

    public func initialize(to value: Pointee, count: Int = 1) {
        var value = value
        !!cudaMemcpy(baseAddress, &value, count * MemoryLayout<Pointee>.stride,
                     cudaMemcpyHostToDevice)
    }

    /// TODO: Change argument type to UnsafeDevicePointer once it's implemented
//    public func initialize(from pointer: UnsafePointer<Pointee>, count: Int) {
//        !!cudaMemcpy(self.pointer, pointer,
//                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
//    }

    public func initialize<C: Collection>(from elements: C)
        where C.Iterator.Element == Pointee, C.IndexDistance == Int {
        ContiguousArray(elements).withUnsafeBufferPointer { ptr in
            !!cudaMemcpy(self.baseAddress, ptr.baseAddress,
                         elements.count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
        }
    }

    public func assign(from pointer: UnsafePointer<Pointee>, count: Int) {
        !!cudaMemcpy(self.baseAddress, pointer,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
    }

    public subscript(i: Int) -> Pointee {
        get {
            return self.advanced(by: i).pointee
        }
        nonmutating set {
            var newValue = newValue
            !!cudaMemcpy(advanced(by: i).baseAddress, &newValue,
                         MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
        }
    }
    
}

