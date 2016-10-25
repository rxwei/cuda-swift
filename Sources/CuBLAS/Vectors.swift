//
//  Vectors.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

import struct CUDARuntime.DeviceArray
import CCuBLAS

public extension BLAS {

    public func sumOfAbsoluteValues(in elements: DeviceArray<Float>) -> Float {
        var data = elements
        var result: Float = 0.0
        data.withUnsafeMutableDevicePointer { dataPtr -> () in
            !!cublasSasum_v2(
                handle,
                Int32(data.count),
                dataPtr.deviceAddress,
                1,
                &result
            )
        }
        return result
    }

    public func sumOfAbsoluteValues(in elements: DeviceArray<Double>) -> Double {
        var data = elements
        var result: Double = 0.0
        data.withUnsafeMutableDevicePointer { dataPtr -> () in
            !!cublasDasum_v2(
                handle,
                Int32(data.count),
                dataPtr.deviceAddress,
                1,
                &result
            )
        }
        return result
    }

    public func add(_ x: DeviceArray<Float>, multipliedBy alpha: Float = 1.0, onto y: inout DeviceArray<Float>) {
        var alpha = alpha
        var x = x
        y.withUnsafeMutableDevicePointer { destPtr -> () in
            x.withUnsafeMutableDevicePointer { srcPtr -> () in
                !!cublasSaxpy_v2(
                    handle,
                    Int32(x.count), &alpha, srcPtr.deviceAddress,
                    1, destPtr.deviceAddress, 1
                )
            }
        }
    }
    
}
