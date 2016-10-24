//
//  BLAS.swift
//  CUDA
//
//  Created by Richard Wei on 9/27/16.
//
//

import CCUDARuntime
import CCuBLAS
@_exported import class CUDARuntime.Stream

public class BLAS {

    let handle: cublasHandle_t

    public static let current = BLAS()

    public init() {
        var handle: cublasHandle_t?
        !!cublasCreate_v2(&handle)
        self.handle = handle!

        /// We will use device pointer reference in this library
        /// for safety and type consistency
        pointerMode = .deviceReference
    }

    deinit {
        cublasDestroy_v2(handle)
    }

    public var version: Int {
        var version: Int32 = 0
        !!cublasGetVersion_v2(handle, &version)
        return Int(version)
    }

    public var stream: Stream? {
        get {
            var streamHandle: cudaStream_t?
            !!cublasGetStream_v2(handle, &streamHandle)
            return streamHandle.flatMap(Stream.current(with:))
        }
        set {
            newValue?.withUnsafeHandle { streamHandle -> () in
                !!cublasSetStream_v2(handle, streamHandle)
            }
        }
    }

    public var allowsAtomics: Bool {
        get {
            var mode: cublasAtomicsMode_t = CUBLAS_ATOMICS_NOT_ALLOWED
            !!cublasGetAtomicsMode(handle, &mode)
            return mode == CUBLAS_ATOMICS_ALLOWED
        }
        set {
            !!cublasSetAtomicsMode(
                handle,
                newValue ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED
            )
        }
    }

    enum PointerMode : UInt32 {
        case deviceReference = 0
        case hostReference = 1
    }

    var pointerMode: PointerMode {
        get {
            var mode: cublasPointerMode_t = CUBLAS_POINTER_MODE_HOST
            !!cublasGetPointerMode_v2(handle, &mode)
            return PointerMode(rawValue: mode.rawValue)!
        }
        set {
            !!cublasSetPointerMode_v2(
                handle,
                cublasPointerMode_t(rawValue: newValue.rawValue)
            )
        }
    }



}
