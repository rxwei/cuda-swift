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
    public func gemm<T: BLASDataProtocol>
        (alpha: Float, 
         A: UnsafeDevicePointer<T>, rowCount m: Int32,
         transpose transposeA: Transpose, leadingDimension lda: Int32,
         B: UnsafeDevicePointer<T>, columnCount n: Int32,
         transpose transposeB: Transpose, leadingDimension ldb: Int32,
         commonDimension k: Int32, beta: Float,
         C: UnsafeMutableDevicePointer<T>, leadingDimension ldc: Int32) {
        var alpha = alpha, beta = beta
        !!cublasGemmEx(
            handle, transposeA.operation, transposeB.operation,
            m, n, k, &alpha,
            A.deviceAddress, T.cType, lda,
            B.deviceAddress, T.cType, ldb,
            &beta,
            C.deviceAddress, T.cType, ldc,
            T.cType, CUBLAS_GEMM_ALGO0
        )
    }

}
