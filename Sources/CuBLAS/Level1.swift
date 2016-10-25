//
//  Level1.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

import CCuBLAS
import struct CUDARuntime.DeviceValue
import struct CUDARuntime.UnsafeMutableDevicePointer
import struct CUDARuntime.UnsafeDevicePointer

public extension BLAS {

    public func sumOfAbsoluteValues(in vector: DeviceVector<Float>) -> DeviceValue<Float> {
        var result = DeviceValue<Float>()
        result.withUnsafeMutableDevicePointer { resultPtr -> () in
            vector.withUnsafeDevicePointer { dataPtr -> () in
                !!cublasSasum_v2(
                    handle,
                    Int32(vector.count),
                    dataPtr.deviceAddress, 1,
                    resultPtr.deviceAddress
                )
            }
        }
        return result
    }

    public func sumOfAbsoluteValues(in vector: DeviceVector<Double>) -> DeviceValue<Double> {
        var result = DeviceValue<Double>()
        result.withUnsafeMutableDevicePointer { resultPtr -> () in
            vector.withUnsafeDevicePointer { dataPtr -> () in
                !!cublasDasum_v2(
                    handle,
                    Int32(vector.count),
                    dataPtr.deviceAddress, 1,
                    resultPtr.deviceAddress
                )
            }
        }
        return result
    }

    public func add(_ x: DeviceVector<Float>,
                    multipliedBy alpha: DeviceValue<Float> = DeviceValue(1.0),
                    onto y: inout DeviceVector<Float>) {
        y.withUnsafeMutableDevicePointer { destPtr -> () in
            x.withUnsafeDevicePointer { srcPtr -> () in
                alpha.withUnsafeDevicePointer { alphaPtr -> () in
                    !!cublasSaxpy_v2(
                        handle,
                        Int32(x.count), alphaPtr.deviceAddress,
                        srcPtr.deviceAddress, 1,
                        destPtr.deviceAddress, 1
                    )
                }
            }
        }
    }

    public func add(_ x: DeviceVector<Double>, multipliedBy alpha: Double = 1.0, onto y: inout DeviceVector<Double>) {
        var alpha = alpha
        y.withUnsafeMutableDevicePointer { destPtr -> () in
            x.withUnsafeDevicePointer { srcPtr -> () in
                !!cublasDaxpy_v2(
                    handle,
                    Int32(x.count), &alpha,
                    srcPtr.deviceAddress, 1,
                    destPtr.deviceAddress, 1
                )
            }
        }
    }
    
    public func copy(_ x: DeviceVector<Float>, onto y: inout DeviceVector<Float>) {
        y.withUnsafeMutableDevicePointer { destPtr in
            x.withUnsafeDevicePointer { srcPtr in
                !!cublasScopy_v2(
                    handle,
                    Int32(x.count),
                    srcPtr.deviceAddress, 1,
                    destPtr.deviceAddress, 1
                )
            }
        }
    }
    
    public func copy(_ x: DeviceVector<Double>, onto y: inout DeviceVector<Double>) {
        y.withUnsafeMutableDevicePointer { destPtr in
            x.withUnsafeDevicePointer { srcPtr in
                !!cublasDcopy_v2(
                    handle,
                    Int32(x.count),
                    srcPtr.deviceAddress, 1,
                    destPtr.deviceAddress, 1
                )
            }
        }
    }

    public func multiply(_ x: inout DeviceVector<Float>, by a: DeviceValue<Float>) {
        x.withUnsafeMutableDevicePointer { destPtr in
            a.withUnsafeDevicePointer { scalarPtr in 
                !!cublasSscal_v2(
                    handle,
                    Int32(x.count),
                    scalarPtr.deviceAddress,
                    destPtr.deviceAddress,
                    1
                )
            }
        }
    }

    public func multiply(_ x: inout DeviceVector<Double>, by a: DeviceValue<Double>) {
        x.withUnsafeMutableDevicePointer { destPtr in
            a.withUnsafeDevicePointer { scalarPtr in 
                !!cublasDscal_v2(
                    handle,
                    Int32(x.count),
                    scalarPtr.deviceAddress,
                    destPtr.deviceAddress,
                    1
                )
            }
        }
    }
    
    public func swap(_ x: inout DeviceVector<Float>, _ y: inout DeviceVector<Float>) {
        x.withUnsafeMutableDevicePointer { xPtr in
            y.withUnsafeMutableDevicePointer { yPtr in
                !!cublasSswap_v2(
                    handle,
                    Int32(x.count),
                    xPtr.deviceAddress, 1,
                    yPtr.deviceAddress, 1
                )
            }
        }
    }
    
    public func swap(_ x: inout DeviceVector<Double>, _ y: inout DeviceVector<Double>) {
        x.withUnsafeMutableDevicePointer { xPtr in
            y.withUnsafeMutableDevicePointer { yPtr in
                !!cublasDswap_v2(
                    handle,
                    Int32(x.count),
                    xPtr.deviceAddress, 1,
                    yPtr.deviceAddress, 1
                )
            }
        }
    }
    
    public func oneBasedIndexOfMin(in vector: DeviceVector<Float>) -> DeviceValue<Int32> {
        var result = DeviceValue<Int32>()
        result.withUnsafeMutableDevicePointer { resultPtr in
            vector.withUnsafeDevicePointer { ptr in
                !!cublasIsamin_v2(
                    handle,
                    Int32(vector.count),
                    ptr.deviceAddress,
                    1,
                    resultPtr.deviceAddress
                )
            }
        }
        return result
    }
    
    public func oneBasedIndexOfMin(in vector: DeviceVector<Double>) -> DeviceValue<Int32> {
        var result = DeviceValue<Int32>()
        result.withUnsafeMutableDevicePointer { resultPtr in
            vector.withUnsafeDevicePointer { ptr in
                !!cublasIdamin_v2(
                    handle,
                    Int32(vector.count),
                    ptr.deviceAddress,
                    1,
                    resultPtr.deviceAddress
                )
            }
        }
        return result
    }
    
    public func oneBasedIndexOfMax(in vector: DeviceVector<Float>) -> DeviceValue<Int32> {
        var result = DeviceValue<Int32>()
        result.withUnsafeMutableDevicePointer { resultPtr in
            vector.withUnsafeDevicePointer { ptr in
                !!cublasIsamax_v2(
                    handle,
                    Int32(vector.count),
                    ptr.deviceAddress,
                    1,
                    resultPtr.deviceAddress
                )
            }
        }
        return result
    }
    
    public func oneBasedIndexOfMax(in vector: DeviceVector<Double>) -> DeviceValue<Int32> {
        var result = DeviceValue<Int32>()
        result.withUnsafeMutableDevicePointer { resultPtr in
            vector.withUnsafeDevicePointer { ptr in
                !!cublasIdamax_v2(
                    handle,
                    Int32(vector.count),
                    ptr.deviceAddress,
                    1,
                    resultPtr.deviceAddress
                )
            }
        }
        return result
    }
    
    /// TODO: These should return DeviceValue. In order to make that possible,
    /// we need to add reference flags in DeviceValue to make it support non-owning
    /// reference.
    //    @inline(__always)
    //    public func min(in vector: DeviceVector<Float>) -> Float {
    //        return vector[indexOfMin(in: vector)]
    //    }
    //
    //    @inline(__always)
    //    public func min(in vector: DeviceVector<Double>) -> Double {
    //        return vector[indexOfMin(in: vector)]
    //    }
    //
    //    @inline(__always)
    //    public func max(in vector: DeviceVector<Float>) -> Float {
    //        return vector[indexOfMax(in: vector)]
    //    }
    //
    //    @inline(__always)
    //    public func max(in vector: DeviceVector<Double>) -> Double {
    //        return vector[indexOfMax(in: vector)]
    //    }
    
    public func dotProduct(_ x: DeviceVector<Float>, _ y: DeviceVector<Float>) -> DeviceValue<Float> {
        var result = DeviceValue<Float>()
        result.withUnsafeMutableDevicePointer { resultPtr in
            x.withUnsafeDevicePointer { xPtr in
                y.withUnsafeDevicePointer { yPtr in
                    !!cublasSdot_v2(
                        handle,
                        Int32(x.count),
                        xPtr.deviceAddress, 1,
                        yPtr.deviceAddress, 1,
                        resultPtr.deviceAddress
                    )
                }
            }
        }
        return result
    }
    
    public func dotProduct(_ x: DeviceVector<Double>, _ y: DeviceVector<Double>) -> DeviceValue<Double> {
        var result = DeviceValue<Double>()
        result.withUnsafeMutableDevicePointer { resultPtr in
            x.withUnsafeDevicePointer { xPtr in
                y.withUnsafeDevicePointer { yPtr in
                    !!cublasDdot_v2(
                        handle,
                        Int32(x.count),
                        xPtr.deviceAddress, 1,
                        yPtr.deviceAddress, 1,
                        resultPtr.deviceAddress
                    )
                }
            }
        }
        return result
    }
    
}

infix operator • : MultiplicationPrecedence

@inline(__always)
public func •(lhs: DeviceVector<Float>, rhs: DeviceVector<Float>) -> DeviceValue<Float> {
    return BLAS.current.dotProduct(lhs, rhs)
}

@inline(__always)
public func •(lhs: DeviceVector<Double>, rhs: DeviceVector<Double>) -> DeviceValue<Double> {
    return BLAS.current.dotProduct(lhs, rhs)
}

@inline(__always)
public func *=(lhs: inout DeviceVector<Float>, rhs: Float) {
    BLAS.current.multiply(&lhs, by: DeviceValue(rhs))
}

@inline(__always)
public func *=(lhs: inout DeviceVector<Double>, rhs: Double) {
    BLAS.current.multiply(&lhs, by: DeviceValue(rhs))
}

@inline(__always)
public func +=(lhs: inout DeviceVector<Float>, rhs: DeviceVector<Float>) {
    BLAS.current.add(rhs, onto: &lhs)
}

@inline(__always)
public func +=(lhs: inout DeviceVector<Double>, rhs: DeviceVector<Double>) {
    BLAS.current.add(rhs, onto: &lhs)
}

@inline(__always)
public func *=(lhs: inout DeviceVector<Float>, rhs: DeviceValue<Float>) {
    BLAS.current.multiply(&lhs, by: rhs)
}

@inline(__always)
public func *=(lhs: inout DeviceVector<Double>, rhs: DeviceValue<Double>) {
    BLAS.current.multiply(&lhs, by: rhs)
}
