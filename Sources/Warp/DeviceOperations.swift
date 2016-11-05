//
//  DeviceOperations.swift
//  CUDA
//
//  Created by Richard Wei on 11/5/16.
//
//

import CuBLAS
import CUDARuntime

public extension DeviceArray where Element : BLASDataProtocol & FloatingPoint {

    public mutating func scale(by alpha: Element) {
        let device = self.device
        let count = self.count
        withUnsafeMutableDevicePointer { ptr in
            BLAS.global(on: device).scale(ptr, stride: 1, count: Int32(count), by: alpha)
        }
    }

    public func scaled(by alpha: Element) -> DeviceArray {
        var copy = self
        copy.scale(by: alpha)
        return copy
    }

    public func reduced() -> Element {
        return withUnsafeDevicePointer { ptr in
            DeviceArray<Element>(repeating: 1, count: self.count, device: self.device)
                .withUnsafeDevicePointer { onePtr in
                BLAS.global(on: self.device).dot(x: ptr, stride: 1, y: onePtr, stride: 1, count: Int32(self.count))
            }
        }
    }

    public func dotProduct(with other: DeviceArray) -> Element {
        return withUnsafeDevicePointer { ptr in
            other.withUnsafeDevicePointer { otherPtr in
                BLAS.global(on: self.device).dot(x: ptr, stride: 1, y: otherPtr, stride: 1,
                                                 count: Int32(Swift.min(self.count, other.count)))
            }
        }
    }

}
