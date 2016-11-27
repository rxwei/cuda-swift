//
//  Level3.swift
//  CUDA
//
//  Created by Richard Wei on 11/11/16.
//
//

import CCuBLAS
import CUDARuntime

public extension BLAS {

    /// Matrix multiplication
    /// C = α transpose(A) transpose(B) + βC
    public func gemm
        (alpha: Float,
         A: UnsafeDevicePointer<Float>, rowCount m: Int32,
         transpose transposeA: Transpose, leadingDimension lda: Int32,
         B: UnsafeDevicePointer<Float>, columnCount n: Int32,
         transpose transposeB: Transpose, leadingDimension ldb: Int32,
         commonDimension k: Int32, beta: Float,
         C: UnsafeMutableDevicePointer<Float>, leadingDimension ldc: Int32) {
        var alpha = alpha, beta = beta
        !!cublasSgemm_v2(
            handle, transposeA.operation, transposeB.operation,
            m, n, k, &alpha, A.deviceAddress, lda, B.deviceAddress, ldb,
            &beta, C.deviceAddress, ldc
        )
    }

    /// Matrix multiplication
    /// C = α transpose(A) transpose(B) + βC
    public func gemm
        (alpha: Double,
         A: UnsafeDevicePointer<Double>, rowCount m: Int32,
         transpose transposeA: Transpose, leadingDimension lda: Int32,
         B: UnsafeDevicePointer<Double>, columnCount n: Int32,
         transpose transposeB: Transpose, leadingDimension ldb: Int32,
         commonDimension k: Int32, beta: Double,
         C: UnsafeMutableDevicePointer<Double>, leadingDimension ldc: Int32) {
        var alpha = alpha, beta = beta
        !!cublasDgemm_v2(
            handle, transposeA.operation, transposeB.operation,
            m, n, k, &alpha, A.deviceAddress, lda, B.deviceAddress, ldb,
            &beta, C.deviceAddress, ldc
        )
    }

}
