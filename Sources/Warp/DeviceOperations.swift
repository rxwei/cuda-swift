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
        return lhs.dotProduct(with: rhs)
    }

}

public extension DeviceArray where Element : KernelDataProtocol {

    public mutating func add(_ other: DeviceArray<Element>, multipliedBy alpha: Element = 1) {
        let axpy = kernelManager.kernel(.axpy, forType: Element.self)
        device.sync {
            let blockCount = (count+127)/128
            try! axpy<<<(blockCount, 128)>>>[.value(alpha), .constantArray(other), .array(&self), .longLong(Int64(count))]
        }
    }

    public func adding(_ other: DeviceArray<Element>, multipliedBy alpha: Element = 1) -> DeviceArray<Element> {
        var copy = self
        copy.add(other, multipliedBy: alpha)
        return copy
    }

    public static func +=(lhs: inout DeviceArray<Element>, rhs: DeviceArray<Element>) {
        return lhs.add(rhs)
    }

    public mutating func scale(by alpha: Element) {
        let scale = kernelManager.kernel(.scale, forType: Element.self)
        device.sync {
            let blockCount = (count+127)/128
            try! scale<<<(blockCount, 128)>>>[.array(&self), .value(alpha), .longLong(Int64(count))]
        }
    }

    @inline(__always)
    public static func *=(lhs: inout DeviceArray<Element>, rhs: Element) {
        lhs.scale(by: rhs)
    }

    public func scaled(by alpha: Element) -> DeviceArray {
        var copy = self
        copy.scale(by: alpha)
        return copy
    }

    public func reduced() -> Element {
        var result = DeviceValue<Element>()
        let sum = kernelManager.kernel(.sum, forType: Element.self)
        device.sync {
            try! sum<<<(1, 1)>>>[.constantArray(self), .longLong(Int64(count)), .valuePointer(&result)]
        }
        return result.value
    }

    public func sumOfAbsoluteValues() -> Element {
        var result = DeviceValue<Element>()
        let asum = kernelManager.kernel(.asum, forType: Element.self)
        device.sync {
            try! asum<<<(1, 1)>>>[.constantArray(self), .longLong(Int64(count)), .valuePointer(&result)]
        }
        return result.value
    }

}
