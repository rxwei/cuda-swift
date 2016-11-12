//
//  Level2.swift
//  CUDA
//
//  Created by Richard Wei on 10/26/16.
//
//

import CCuBLAS
import struct CUDARuntime.UnsafeMutableDevicePointer
import struct CUDARuntime.UnsafeDevicePointer

public extension BLAS {

    public enum Transpose : UInt32 {
        case none = 0
        case transpose = 1
        case conjugateTranspose = 2

        var operation: cublasOperation_t {
            return cublasOperation_t(rawValue)
        }
    }

    public func gemv
        (alpha: Float,
         A: UnsafeDevicePointer<Float>, rowCount m: Int32, columnCount n: Int32,
         transpose: Transpose, leadingDimension lda: Int32,
         x: UnsafeDevicePointer<Float>, stride strideX: Int32,
         beta: Float,
         y: UnsafeMutableDevicePointer<Float>, stride strideY: Int32) {
        var alpha = alpha, beta = beta
        !!cublasSgemv_v2(
            handle, transpose.operation, m, n, &alpha,
            A.deviceAddress, lda, x.deviceAddress, strideX,
            &beta, y.deviceAddress, strideY
        )
    }

    public func gemv
        (alpha: Double,
         A: UnsafeDevicePointer<Double>, rowCount m: Int32, columnCount n: Int32,
         transpose: Transpose, leadingDimension lda: Int32,
         x: UnsafeDevicePointer<Double>, stride strideX: Int32,
         beta: Double,
         y: UnsafeMutableDevicePointer<Double>, stride strideY: Int32) {
        var alpha = alpha, beta = beta
        !!cublasDgemv_v2(
            handle, transpose.operation, m, n, &alpha,
            A.deviceAddress, lda, x.deviceAddress, strideX,
            &beta, y.deviceAddress, strideY
        )
    }

}
