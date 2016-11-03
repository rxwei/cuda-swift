//
//  Level1.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

import CCuBLAS
import struct CUDARuntime.UnsafeMutableDevicePointer
import struct CUDARuntime.UnsafeDevicePointer

public extension BLAS {
    
    public func asum(_ vector: UnsafeDevicePointer<Float>, stride: Int, count: Int) -> Float {
        var result: Float = 0.0
        !!cublasSasum_v2(handle, Int32(count), vector.deviceAddress, Int32(stride), &result)
        return result
    }
    
    public func asum(_ vector: UnsafeDevicePointer<Double>, stride: Int, count: Int) -> Double {
        var result: Double = 0.0
        !!cublasDasum_v2(handle, Int32(count), vector.deviceAddress, Int32(stride), &result)
        return result
    }

    public func axpy(alpha: Float = 0.0,
                     x: UnsafeDevicePointer<Float>, stride strideX: Int,
                     to y: UnsafeMutableDevicePointer<Float>, stride strideY: Int,
                     count: Int) {
        var alpha = alpha
        !!cublasSaxpy_v2(handle, Int32(count), &alpha,
                         x.deviceAddress, Int32(strideX),
                         y.deviceAddress, Int32(strideY))
    }
    
    public func axpy(alpha: Double = 0.0,
                     x: UnsafeDevicePointer<Double>, stride strideX: Int,
                     to y: UnsafeMutableDevicePointer<Double>, stride strideY: Int,
                     count: Int) {
        var alpha = alpha
        !!cublasDaxpy_v2(handle, Int32(count), &alpha,
                         x.deviceAddress, Int32(strideX),
                         y.deviceAddress, Int32(strideY))
    }
    
    public func copy(_ x: UnsafeDevicePointer<Float>, stride strideX: Int,
                     to y: UnsafeMutableDevicePointer<Float>, stride strideY: Int,
                     count: Int) {
        !!cublasScopy_v2(handle, Int32(count),
                         x.deviceAddress, Int32(strideX),
                         y.deviceAddress, Int32(strideY))
    }

    public func copy(_ x: UnsafeDevicePointer<Double>, stride strideX: Int,
                     to y: UnsafeMutableDevicePointer<Double>, stride strideY: Int,
                     count: Int) {
        !!cublasDcopy_v2(handle, Int32(count),
                         x.deviceAddress, Int32(strideX),
                         y.deviceAddress, Int32(strideY))
    }

    public func scal(_ x: UnsafeMutableDevicePointer<Float>, stride: Int, count: Int,
                     by alpha: UnsafeDevicePointer<Float>) {
        !!cublasSscal_v2(handle, Int32(count),
                         alpha.deviceAddress, x.deviceAddress,
                         Int32(stride))
    }
    
    public func scal(_ x: UnsafeMutableDevicePointer<Double>, stride: Int, count: Int,
                     by alpha: UnsafeDevicePointer<Double>) {
        !!cublasDscal_v2(handle, Int32(count),
                         alpha.deviceAddress, x.deviceAddress,
                         Int32(stride))
    }

    public func swap(_ x: inout UnsafeMutableDevicePointer<Float>, stride strideX: Int,
                     _ y: inout UnsafeMutableDevicePointer<Float>, stride strideY: Int,
                     count: Int) {
        !!cublasSswap_v2(handle, Int32(count),
                         x.deviceAddress, Int32(strideX),
                         y.deviceAddress, Int32(strideY))
    }
    
    public func swap(_ x: inout UnsafeMutableDevicePointer<Double>, stride strideX: Int,
                     _ y: inout UnsafeMutableDevicePointer<Double>, stride strideY: Int,
                     count: Int) {
        !!cublasDswap_v2(handle, Int32(count),
                         x.deviceAddress, Int32(strideX),
                         y.deviceAddress, Int32(strideY))
    }
    
    public func iamin(_ vector: UnsafeDevicePointer<Float>, stride: Int, count: Int) -> Int {
        var result: Int32 = 0
        !!cublasIsamin_v2(handle, Int32(count),
                          vector.deviceAddress, Int32(stride),
                          &result
        )
        return Int(result)
    }

    public func iamin(_ vector: UnsafeDevicePointer<Double>, stride: Int, count: Int) -> Int {
        var result: Int32 = 0
        !!cublasIdamin_v2(handle, Int32(count),
                          vector.deviceAddress, Int32(stride),
                          &result
        )
        return Int(result)
    }

    public func iamax(_ vector: UnsafeDevicePointer<Float>, stride: Int, count: Int) -> Int {
        var result: Int32 = 0
        !!cublasIsamax_v2(handle, Int32(count),
                          vector.deviceAddress, Int32(stride),
                          &result
        )
        return Int(result)
    }
    
    public func iamax(_ vector: UnsafeDevicePointer<Double>, stride: Int, count: Int) -> Int {
        var result: Int32 = 0
        !!cublasIdamax_v2(handle, Int32(count),
                          vector.deviceAddress, Int32(stride),
                          &result
        )
        return Int(result)
    }

    public func dot(_ x: UnsafeDevicePointer<Float>, stride strideX: Int,
                    _ y: UnsafeDevicePointer<Float>, stride strideY: Int, count: Int) -> Float {
        var result: Float = 0.0
        !!cublasSdot_v2(handle, Int32(count),
                        x.deviceAddress, Int32(strideX),
                        y.deviceAddress, Int32(strideY),
                        &result)
        return result
    }

    public func dot(_ x: UnsafeDevicePointer<Double>, stride strideX: Int,
                    _ y: UnsafeDevicePointer<Double>, stride strideY: Int, count: Int) -> Double {
        var result: Double = 0.0
        !!cublasDdot_v2(handle, Int32(count),
                        x.deviceAddress, Int32(strideX),
                        y.deviceAddress, Int32(strideY),
                        &result)
        return result
    }
    
}
