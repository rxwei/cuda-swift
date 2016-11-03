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
    }

}
