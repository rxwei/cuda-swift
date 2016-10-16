//
//  BLAS.swift
//  SAD
//
//  Created by Richard Wei on 9/27/16.
//
//

import CCUDARuntime
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

    public func absSum(_ data: [Float]) -> Float {
        var devPtr: UnsafeMutableRawPointer?
        cudaMalloc(&devPtr, data.count * MemoryLayout<Float>.stride)
        data.withUnsafeBufferPointer { dataBuffer -> () in
            cudaMemcpy(devPtr, dataBuffer.baseAddress,
                       data.count * MemoryLayout<Float>.stride,
                       cudaMemcpyHostToDevice)
        }
        var result: Float = 0.0
        cublasSasum_v2(
            handle,
            Int32(data.count),
            unsafeBitCast(devPtr, to: UnsafePointer<Float>.self),
            1,
            &result
        )
        cudaFree(devPtr)
        return result
    }

    public func absSum(_ data: [Double]) -> Double {
        var devPtr: UnsafeMutableRawPointer?
        cudaMalloc(&devPtr, data.count * MemoryLayout<Double>.stride)
        data.withUnsafeBufferPointer { dataBuffer -> () in
            cudaMemcpy(devPtr, dataBuffer.baseAddress,
                       data.count * MemoryLayout<Double>.stride,
                       cudaMemcpyHostToDevice)
        }
        var result: Double = 0.0
        cublasDasum_v2(
            handle,
            Int32(data.count),
            unsafeBitCast(devPtr, to: UnsafePointer<Double>.self),
            1,
            &result
        )
        cudaFree(devPtr)
        return result
    }
    
}
