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

    public func asum(_ vector: UnsafeDevicePointer<Float>, stride: Int32, count: Int32) -> Float {
        var result: Float = 0.0
        !!cublasSasum_v2(handle, count, vector.deviceAddress, stride, &result)
        return result
    }
    
    public func asum(_ vector: UnsafeDevicePointer<Double>, stride: Int32, count: Int32) -> Double {
        var result: Double = 0.0
        !!cublasDasum_v2(handle, count, vector.deviceAddress, stride, &result)
        return result
    }

    public func axpy<T: BLASDataProtocol>(alpha: T,
                     x: UnsafeDevicePointer<T>, stride strideX: Int32,
                     y: UnsafeMutableDevicePointer<T>, stride strideY: Int32,
                     count: Int32) {
        var alpha = alpha
        !!cublasAxpyEx(handle, count, &alpha, T.cType,
                     x.deviceAddress, T.cType, strideX,
                     y.deviceAddress, T.cType, strideY, T.cType)
    }
    
    public func copy(_ vector: UnsafeDevicePointer<Float>, stride strideX: Int32,
                     to y: UnsafeMutableDevicePointer<Float>, stride strideY: Int32,
                     count: Int32) {
        !!cublasScopy_v2(handle, count,
                         vector.deviceAddress, strideX,
                         y.deviceAddress, strideY)
    }
    
    public func copy(_ vector: UnsafeDevicePointer<Double>, stride strideX: Int32,
                     to y: UnsafeMutableDevicePointer<Double>, stride strideY: Int32,
                     count: Int32) {
        !!cublasDcopy_v2(handle, count,
                         vector.deviceAddress, strideX,
                         y.deviceAddress, strideY)
    }

    public func scal<T: BLASDataProtocol>(
        _ x: UnsafeMutableDevicePointer<T>, stride: Int32, count: Int32, alpha: T) {
        var alpha = alpha
        !!cublasScalEx(handle, count, &alpha, T.cType, x.deviceAddress, T.cType, stride, T.cType)
    }
    
    public func swap(_ x: inout UnsafeMutableDevicePointer<Float>, stride strideX: Int32,
                     _ y: inout UnsafeMutableDevicePointer<Float>, stride strideY: Int32,
                     count: Int32) {
        !!cublasSswap_v2(handle, count,
                         x.deviceAddress, strideX,
                         y.deviceAddress, strideY)
    }
    
    public func swap(_ x: inout UnsafeMutableDevicePointer<Double>, stride strideX: Int32,
                     _ y: inout UnsafeMutableDevicePointer<Double>, stride strideY: Int32,
                     count: Int32) {
        !!cublasDswap_v2(handle, count,
                         x.deviceAddress, strideX,
                         y.deviceAddress, strideY)
    }
    
    public func iamin(_ vector: UnsafeDevicePointer<Float>, stride: Int32, count: Int32) -> Int32 {
        var result: Int32 = 0
        !!cublasIsamin_v2(handle, count, vector.deviceAddress, stride, &result)
        return Int32(result)
    }
    
    public func iamin(_ vector: UnsafeDevicePointer<Double>, stride: Int32, count: Int32) -> Int32 {
        var result: Int32 = 0
        !!cublasIdamin_v2(handle, count, vector.deviceAddress, stride, &result)
        return Int32(result)
    }
    
    public func iamax(_ vector: UnsafeDevicePointer<Float>, stride: Int32, count: Int32) -> Int32 {
        var result: Int32 = 0
        !!cublasIsamax_v2(handle, count, vector.deviceAddress, stride, &result)
        return Int32(result)
    }
    
    public func iamax(_ vector: UnsafeDevicePointer<Double>, stride: Int32, count: Int32) -> Int32 {
        var result: Int32 = 0
        !!cublasIdamax_v2(handle, count, vector.deviceAddress, stride, &result)
        return Int32(result)
    }

    public func dot<T: BLASDataProtocol>(
        x: UnsafeDevicePointer<T>, stride strideX: Int32,
        y: UnsafeDevicePointer<T>, stride strideY: Int32, count: Int32) -> T {
        var result: T!
        !!cublasDotEx(handle, count,
                      x.deviceAddress, T.cType, strideX,
                      y.deviceAddress, T.cType, strideY,
                      &result, T.cType, T.cType)
        return result
    }

}
