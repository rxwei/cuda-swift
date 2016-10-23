//
//  BLAS.swift
//  SAD
//
//  Created by Richard Wei on 9/27/16.
//
//

import CUDARuntime
import CCuBLAS

public class BLAS {

    private var handle: cublasHandle_t

    public static let global = BLAS()

    public init() {
        var handle: cublasHandle_t?
        guard cublasCreate_v2(&handle) == CUBLAS_STATUS_SUCCESS else {
            fatalError("cuBLAS unavailable") /// Oops!
        }
        self.handle = handle!
    }

    deinit {
        cublasDestroy_v2(handle)
    }

    public func absSum(_ data: DeviceArray<Float>) -> Float {
        var data = data
        var result: Float = 0.0
        data.withUnsafeMutableDevicePointer { dataPtr -> () in
            cublasSasum_v2(
                handle,
                Int32(data.count),
                dataPtr.deviceAddress,
                1,
                &result
            )
        }
        return result
    }

    public func absSum(_ data: DeviceArray<Double>) -> Double {
        var data = data
        var result: Double = 0.0
        data.withUnsafeMutableDevicePointer { dataPtr -> () in
            cublasDasum_v2(
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
                cublasSaxpy_v2(
                    handle,
                    Int32(x.count), &alpha, srcPtr.deviceAddress,
                    1, destPtr.deviceAddress, 1
                )
            }
        }
    }
    
}
