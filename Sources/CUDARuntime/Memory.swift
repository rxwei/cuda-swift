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
public struct UnsafeMutableDevicePointer<Pointee> {

    public var pointer: UnsafeMutablePointer<Pointee>?

    internal init(rawPointer: UnsafeMutableRawPointer) {
        self.pointer = rawPointer.assumingMemoryBound(to: Pointee.self)
    }

    public static func allocate(capacity: Int) -> UnsafeMutableDevicePointer<Pointee> {
        var pointer: UnsafeMutableRawPointer?
        !!cudaMalloc(&pointer, capacity * MemoryLayout<Pointee>.size)
        return UnsafeMutableDevicePointer(rawPointer: pointer!)
    }

    public func deallocate() {
        !!cudaFree(pointer)
    }

    public func initialize(to value: Int32, count: Int = 1) {
        !!cudaMemset(pointer, value, count)
    }
    
    public func initialize(to value: Float32, count: Int = 1) {
        !!cudaMemset(pointer, unsafeBitCast(value, to: Int32.self), count)
    }

    public func initialize(to value: Pointee, count: Int = 1) {
        var value = value
        !!cudaMemcpy(pointer, &value, count * MemoryLayout<Pointee>.stride,
                     cudaMemcpyHostToDevice)
    }

    public func initialize(from pointer: UnsafePointer<Pointee>, count: Int) {
        !!cudaMemcpy(self.pointer, pointer,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
    }

    public func initialize<C: Collection>(from elements: C)
        where C.Iterator.Element == Pointee, C.IndexDistance == Int {
        ContiguousArray(elements).withUnsafeBufferPointer { ptr in
            !!cudaMemcpy(self.pointer, ptr.baseAddress,
                         elements.count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
        }
    }

    public func assign(from pointer: UnsafePointer<Pointee>, count: Int) {
        !!cudaMemcpy(self.pointer, pointer,
                     count * MemoryLayout<Pointee>.stride, cudaMemcpyHostToDevice)
    }

    public var pointee: Pointee {
        var pointee: Pointee?
        !!cudaMemcpy(&pointee, pointer, MemoryLayout<Pointee>.size, cudaMemcpyDeviceToHost)
        return pointee!
    }

}
