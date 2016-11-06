//
//  DeviceOperations.swift
//  CUDA
//
//  Created by Richard Wei on 11/5/16.
//
//

import CuBLAS
import CUDARuntime

infix operator • : MultiplicationPrecedence

public extension DeviceArray where Element : BLASDataProtocol & FloatingPoint {

    public mutating func add(_ other: DeviceArray<Element>, multipliedBy alpha: Element = 0) {
        withUnsafeMutableDevicePointer { ptr in
            other.withUnsafeDevicePointer { otherPtr in
                BLAS.global(on: self.device).axpy(
                    alpha: alpha,
                    x: otherPtr, stride: 1,
                    y: ptr, stride: 1,
                    count: Int32(Swift.min(self.count, other.count))
                )
            }
        }
    }

    public func adding(_ other: DeviceArray<Element>, multipliedBy alpha: Element = 0) -> DeviceArray<Element> {
        var copy = self
        copy.add(other, multipliedBy: alpha)
        return copy
    }

    @inline(__always)
    public static func +=(lhs: inout DeviceArray<Element>, rhs: DeviceArray<Element>) -> DeviceArray<Element> {
        return lhs.adding(rhs)
    }

    public mutating func scale(by alpha: Element) {
        withUnsafeMutableDevicePointer { ptr in
            BLAS.global(on: self.device).scale(ptr, stride: 1, count: Int32(self.count), by: alpha)
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

    public func dotProduct(with other: DeviceArray) -> Element {
        return withUnsafeDevicePointer { ptr in
            other.withUnsafeDevicePointer { otherPtr in
                BLAS.global(on: self.device).dot(x: ptr, stride: 1, y: otherPtr, stride: 1,
                                                 count: Int32(Swift.min(self.count, other.count)))
            }
        }
    }

    @inline(__always)
    public static func •(lhs: DeviceArray<Element>, rhs: DeviceArray<Element>) -> Element {
        return lhs.dotProduct(with: rhs)
    }

}

public extension DeviceArray where Element : KernelDataProtocol {

    public func reduced() -> Element {
        var copy = self
        var result = DeviceValue<Element>()
        KernelManager.global(on: device).launchKernel(
            .sum, forType: Element.self,
            arguments: [.array(&copy), .longLong(Int64(count)), .value(&result)],
            blockCount: 1, threadCount: 1
        )
        return result.value
    }

    public func sumOfAbsoluteValues() -> Element {
        var copy = self
        var result = DeviceValue<Element>()
        KernelManager.global(on: device).launchKernel(
            .asum, forType: Element.self,
            arguments: [.array(&copy), .longLong(Int64(count)), .value(&result)],
            blockCount: 1, threadCount: 1
        )
        return result.value
    }

}
