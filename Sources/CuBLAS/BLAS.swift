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

    private var handle: cublasHandle_t?

    public static let global = BLAS()

    public init() {
        guard cublasCreate_v2(&handle) == CUBLAS_STATUS_SUCCESS else {
            fatalError("cuBLAS unavailable") /// Oops!
        }
    }

    deinit {
        cublasDestroy_v2(handle)
    }

    public func absSum(_ data: DeviceArray<Float>) -> Float {
        var result: Float = 0.0
        data.withUnsafeDevicePointer { ptr -> () in
            cublasSasum_v2(
                handle,
                Int32(data.count),
                ptr,
                1,
                &result
            )
        }
        return result
    }

    public func absSum(_ data: DeviceArray<Double>) -> Double {
        var result: Double = 0.0
        data.withUnsafeDevicePointer { ptr -> () in
            cublasDasum_v2(
                handle,
                Int32(data.count),
                ptr,
                1,
                &result
            )
        }
        return result
    }
    
}
