//
//  DeviceOperations.swift
//  CUDA
//
//  Created by Richard Wei on 11/5/16.
//
//

import CuBLAS
import CUDADriver

infix operator • : MultiplicationPrecedence

extension DeviceArray {
    var kernelManager: KernelManager {
        return KernelManager.shared(on: device)
    }
}

public extension DeviceArray where Element : BLASDataProtocol & FloatingPoint {

    public func dotProduct(with other: DeviceArray) -> Element {
        return withUnsafeDevicePointer { ptr in
            other.withUnsafeDevicePointer { otherPtr in
                BLAS.global(on: self.device).dot(
                    x: ptr, stride: 1, y: otherPtr, stride: 1,
                    count: Int32(Swift.min(self.count, other.count))
                )
            }
        }
    }

    @inline(__always)
    public static func •(lhs: DeviceArray<Element>, rhs: DeviceArray<Element>) -> Element {
        precondition(lhs.count == rhs.count,
                     "Two arrays have different number of elements");
        return lhs.dotProduct(with: rhs)
    }

}

public extension DeviceArray where Element : KernelDataProtocol {

    /// Add each element from the other array to each element of this array.
    ///
    /// - Parameters:
    ///   - other: the other array
    ///   - alpha: multiplication factor to the other array before adding
    public mutating func formAddition(with other: DeviceArray<Element>,
                                      multipliedBy alpha: Element = 1) {
        let axpy = kernelManager.kernel(.axpy, forType: Element.self)
        let blockSize = Swift.min(128, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! axpy<<<(blockCount, blockSize)>>>[
                .value(alpha),
                .constPointer(to: other),
                .constPointer(to: self),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }
    
    public mutating func multiplyElements(by alpha: Element) {
        let scale = kernelManager.kernel(.scale, forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! scale<<<(blockCount, blockSize)>>>[
                .value(alpha),
                .constPointer(to: self),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    public mutating func formElementwise(_ operation: BinaryKernelOperation,
                                         with other: DeviceArray<Element>) {
        let elementOp = kernelManager.kernel(.elementwise, operation: operation, forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! elementOp<<<(blockCount, blockSize)>>>[
                .constPointer(to: self),
                .constPointer(to: other),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    public mutating func formElementwiseResult(
        operation: BinaryKernelOperation, x: DeviceArray<Element>, y: DeviceArray<Element>) {
        let elementOp = kernelManager.kernel(.elementwise, operation: operation, forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! elementOp<<<(blockCount, blockSize)>>>[
                .constPointer(to: x),
                .constPointer(to: y),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    public func sum() -> Element {
        var result = DeviceValue<Element>()
        let sum = kernelManager.kernel(.sum, forType: Element.self)
        device.sync {
            try! sum<<<(1, 1)>>>[
                .constPointer(to: self), .longLong(Int64(count)), .pointer(to: &result)
            ]
        }
        return result.value
    }

    public func sumOfAbsoluteValues() -> Element {
        var result = DeviceValue<Element>()
        let asum = kernelManager.kernel(.asum, forType: Element.self)
        device.sync {
            try! asum<<<(1, 1)>>>[
                .constPointer(to: self), .longLong(Int64(count)), .pointer(to: &result)
            ]
        }
        return result.value
    }

    public mutating func fill(with element: Element) {
        /// If type is 32-bit, then perform memset
        if MemoryLayout<Element>.stride == 4 {
            withUnsafeMutableDevicePointer { ptr in
                ptr.assign(element, count: self.count)
            }
        }
        /// For arbitrary type, perform kernel operation
        let fill = kernelManager.kernel(.fill, forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! fill<<<(blockSize, blockCount)>>>[
                .pointer(to: &self), .value(element), .longLong(Int64(count))
            ]
        }
    }

}

public extension DeviceArray where Element : KernelDataProtocol & FloatingPoint {

    public mutating func transform(by functor: FloatingPointKernelFunctor) {
        let transformer = kernelManager.kernel(.transform, functor: functor, forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! transformer<<<(blockCount, blockSize)>>>[
                .constPointer(to: self), .pointer(to: &self), .longLong(Int64(count))
            ]
        }
    }
    
}
