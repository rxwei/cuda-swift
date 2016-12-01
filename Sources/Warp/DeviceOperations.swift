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
        precondition(count == other.count, "Array count mismatch");
        return BLAS.global(on: self.device).dot(
            x: self.unsafeDevicePointer, stride: 1, y: other.unsafeDevicePointer,
            stride: 1, count: Int32(Swift.min(self.count, other.count))
        )
    }

    @inline(__always)
    public static func •(lhs: DeviceArray<Element>, rhs: DeviceArray<Element>) -> Element {
        return lhs.dotProduct(with: rhs)
    }

}

public extension DeviceArray where Element : KernelDataProtocol {

    public mutating func addElements(by x: Element) {
        let scalarRight = kernelManager.kernel(.scalarRight,
                                               operation: .addition,
                                               forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! scalarRight<<<(blockCount, blockSize)>>>[
                .value(1 as Element),
                .constPointer(to: self),
                .value(x),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    public mutating func subtractElements(by x: Element) {
        let scalarRight = kernelManager.kernel(.scalarRight,
                                               operation: .subtraction,
                                               forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! scalarRight<<<(blockCount, blockSize)>>>[
                .value(1 as Element),
                .constPointer(to: self),
                .value(x),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    public mutating func multiplyElements(by x: Element) {
        let scalarRight = kernelManager.kernel(.scalarRight,
                                               operation: .multiplication,
                                               forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! scalarRight<<<(blockCount, blockSize)>>>[
                .value(1 as Element),
                .constPointer(to: self),
                .value(x),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    public mutating func divideElements(by x: Element) {
        let scalarRight = kernelManager.kernel(.scalarRight,
                                               operation: .division,
                                               forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! scalarRight<<<(blockCount, blockSize)>>>[
                .value(1 as Element),
                .constPointer(to: self),
                .value(x),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    /// Assign from elements of the other array multiplied by a constant
    ///
    /// - Parameters:
    ///   - other: array to assign from
    ///   - alpha: mulplication constant
    public mutating func assign(from other: DeviceArray<Element>,
                                multipliedBy alpha: Element = 1) {
        let count = Swift.min(self.count, other.count)
        let scale = kernelManager.kernel(.scalarRight,
                                         operation: .multiplication,
                                         forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! scale<<<(blockCount, blockSize)>>>[
                .value(1 as Element),
                .constPointer(to: self),
                .value(alpha),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    /// Form elementwise operation result with the other array
    ///
    /// - Parameters:
    ///   - operation: binary operation to perform on self and other
    ///   - other: the other array
    public mutating func formElementwise(_ operation: DeviceBinaryOperation,
                                         with other: DeviceArray<Element>,
                                         multipliedBy alpha: Element = 1) {
        precondition(count == other.count, "Array count mismatch")
        let elementOp = kernelManager.kernel(.elementwise,
                                             operation: operation,
                                             forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! elementOp<<<(blockCount, blockSize)>>>[
                .value(1 as Element),
                .constPointer(to: self),
                .value(alpha),
                .constPointer(to: other),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    /// Assign result of elementwise operation on x and y to self
    ///
    /// - Parameters:
    ///   - operation: binary operation to perform on x and y
    ///   - x: arary x
    ///   - y: array y
    public mutating func assignResult(of operation: DeviceBinaryOperation,
                                      left: DeviceArray<Element>,
                                      multipliedBy alpha: Element,
                                      right: Element) {
        let count = Swift.min(self.count, left.count)
        let elementOp = kernelManager.kernel(.scalarRight,
                                             operation: operation,
                                             forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! elementOp<<<(blockCount, blockSize)>>>[
                .value(alpha),
                .constPointer(to: left),
                .value(right),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    /// Assign αx <OP> βy to self
    ///
    /// - Parameters:
    ///   - operation: binary operation to be performed on αx and βy 
    ///   - x: left-hand side array
    ///   - alpha: constant to multiply the left-hand side by
    ///   - y: right-hand side array
    ///   - beta: constant to multiply the right-hand side by
    public mutating func assignElementwiseResult(of operation: DeviceBinaryOperation,
                                                 left: DeviceArray<Element>, multipliedBy alpha: Element = 1,
                                                 right: DeviceArray<Element>, multipliedBy beta: Element = 1) {
        let count = Swift.min(self.count, left.count, right.count)
        let elementOp = kernelManager.kernel(.elementwise, operation: operation, forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! elementOp<<<(blockCount, blockSize)>>>[
                .value(alpha),
                .constPointer(to: left),
                .value(beta),
                .constPointer(to: right),
                .pointer(to: &self),
                .longLong(Int64(count))
            ]
        }
    }

    /// Compute the sum of all elements
    ///
    /// - Returns: sum
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

    /// Compute the sum of absolute values of each element
    ///
    /// - Returns: sum
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

    /// Fill the array with specified value
    ///
    /// - Parameter value: value to fill with
    public mutating func fill(with value: Element) {
        let fill = kernelManager.kernel(.fill, forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! fill<<<(blockSize, blockCount)>>>[
                .pointer(to: &self), .value(value), .longLong(Int64(count))
            ]
        }
    }

}

public extension DeviceArray where Element : KernelDataProtocol & FloatingPoint {

    /// Assign the other array with transformation to self
    ///
    /// - Parameters:
    ///   - other: other array to transform and subsequently to assign from
    ///   - transformation: transformation to apply to the other array
    public mutating func assign(from other: DeviceArray<Element>,
                                transformedBy transformation: DeviceUnaryTransformation) {
        let count = Swift.min(self.count, other.count)
        let transformer = kernelManager.kernel(.transform,
                                               transformation: transformation,
                                               forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! transformer<<<(blockCount, blockSize)>>>[
                .constPointer(to: other), .pointer(to: &self), .longLong(Int64(count))
            ]
        }
    }

    /// Transform self in place
    ///
    /// - Parameter transformation: transformation to apply
    public mutating func transform(by transformation: DeviceUnaryTransformation) {
        let transformer = kernelManager.kernel(.transform, 
                                               transformation: transformation,
                                               forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! transformer<<<(blockCount, blockSize)>>>[
                .constPointer(to: self), .pointer(to: &self), .longLong(Int64(count))
            ]
        }
    }
    
}
