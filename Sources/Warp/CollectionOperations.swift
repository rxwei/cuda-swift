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

extension DeviceCollection {
    var kernelManager: KernelManager {
        return KernelManager.shared(on: device)
    }
}

public enum UnaryOperation : StaticString, SourceHashable {
    case exp = "exp"
    case log = "log"
    case logb = "logb"
    case log2 = "log2"
    case log10 = "log10"
    case log1p = "log1p"
    case cos = "cos"
    case cospi = "cospi"
    case sin = "sin"
    case sinpi = "sinpi"
    case tan = "tan"
    case tgamma = "tgamma"
    case trunc = "trunc"
    case sqrt = "sqrt"
    case cbrt = "cbrt"
    case tanh = "tanh"
    case sinh = "sinh"
    case cosh = "cosh"
    case acos = "acos"
    case asin = "asin"
    case atan = "atan"
    case floor = "floor"
    case ceil = "ceil"
    case round = "round"
    case rsqrt = "rsqrt"
    case rcbrt = "rcbrt"
    case rint = "rint"

    func source<T: FloatingPoint>(forType _: T.Type) -> String {
        let baseName = self.source
        if T.self == Float.self {
            return baseName + "f"
        }
        return baseName
    }
}

public enum BinaryOperation : SourceHashable {
    case addition
    case subtraction
    case multiplication
    case division

    var source: String {
        switch self {
        case .addition: return "((_x_) + (_y_))"
        case .subtraction: return "((_x_) - (_y_))"
        case .multiplication: return "((_x_) * (_y_))"
        case .division: return "((_x_) / (_y_))"
        }
    }
}

public extension DeviceCollection where Element : BLASDataProtocol & FloatingPoint, Index == Int, IndexDistance == Int {

    public func dotProduct(with other: Self) -> Element {
        precondition(count == other.count, "Array count mismatch");
        return BLAS.global(on: self.device).dot(
            x: self.unsafeDevicePointer, stride: 1, y: other.unsafeDevicePointer,
            stride: 1, count: Int32(Swift.min(self.count, other.count))
        )
    }

    @inline(__always)
    public static func •(lhs: Self, rhs: Self) -> Element {
        return lhs.dotProduct(with: rhs)
    }

}

public extension MutableDeviceCollection where Element : KernelDataProtocol, Index == Int, IndexDistance == Int {

    public mutating func incrementElements(by x: Element) {
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
                .longLong(Int64(count)),
                .pointer(to: &self)
            ]
        }
    }

    public mutating func decrementElements(by x: Element) {
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
                .longLong(Int64(count)),
                .pointer(to: &self)
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
                .longLong(Int64(count)),
                .pointer(to: &self)
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
                .longLong(Int64(count)),
                .pointer(to: &self)
            ]
        }
    }

    /// Assign from elements of the other array multiplied by a constant
    ///
    /// - Parameters:
    ///   - other: array to assign from
    ///   - alpha: mulplication constant
    public mutating func assign(from other: Self,
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
                .longLong(Int64(count)),
                .pointer(to: &self)
            ]
        }
    }

    /// Form elementwise operation result with the other array
    ///
    /// - Parameters:
    ///   - operation: binary operation to perform on self and other
    ///   - other: the other array
    public mutating func formElementwise(_ operation: BinaryOperation,
                                         with other: Self,
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
                .longLong(Int64(count)),
                .pointer(to: &self)
            ]
        }
    }

    public mutating func assign(from operation: BinaryOperation,
                                left: Self,
                                multipliedBy alpha: Element = 1,
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
                .longLong(Int64(count)),
                .pointer(to: &self)
            ]
        }
    }

    public mutating func assign(from operation: BinaryOperation,
                                left: Element,
                                right: Self,
                                multipliedBy alpha: Element = 1) {
        let count = Swift.min(self.count, right.count)
        let elementOp = kernelManager.kernel(.scalarLeft,
                                             operation: operation,
                                             forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! elementOp<<<(blockCount, blockSize)>>>[
                .value(left),
                .value(alpha),
                .constPointer(to: right),
                .longLong(Int64(count)),
                .pointer(to: &self)
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
    public mutating func assign(from operation: BinaryOperation,
                                left: Self, multipliedBy alpha: Element = 1,
                                right: Self, multipliedBy beta: Element = 1) {
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
                .longLong(Int64(count)),
                .pointer(to: &self)
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

public extension MutableDeviceCollection where Element : KernelDataProtocol & FloatingPoint, Index == Int, IndexDistance == Int {

    /// Assign the other array with transformation to self
    ///
    /// - Parameters:
    ///   - other: other array to transform and subsequently to assign from
    ///   - transformation: transformation to apply to the other array
    public mutating func assign(from other: Self,
                                transformedBy transformation: UnaryOperation) {
        let count = Swift.min(self.count, other.count)
        let transformer = kernelManager.kernel(.transform,
                                               transformation: transformation,
                                               forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! transformer<<<(blockCount, blockSize)>>>[
                .constPointer(to: other), .longLong(Int64(count)), .pointer(to: &self)
            ]
        }
    }

    /// Transform self in place
    ///
    /// - Parameter transformation: transformation to apply
    public mutating func transform(by transformation: UnaryOperation) {
        let transformer = kernelManager.kernel(.transform, 
                                               transformation: transformation,
                                               forType: Element.self)
        let blockSize = Swift.min(512, count)
        let blockCount = (count+blockSize-1)/blockSize
        device.sync {
            try! transformer<<<(blockCount, blockSize)>>>[
                .constPointer(to: self), .longLong(Int64(count)), .pointer(to: &self)
            ]
        }
    }

}
