//
//  BLAS.swift
//  CUDA
//
//  Created by Richard Wei on 9/27/16.
//
//

import CCuBLAS
import typealias CCUDARuntime.cudaStream_t

@_exported import class CUDARuntime.Stream
import struct CUDARuntime.DeviceArray

public typealias DeviceVector<T> = DeviceArray<T>
@_exported import struct CUDARuntime.DeviceValue

open class BLAS {

    let handle: cublasHandle_t

    open static let main = BLAS()

    /// Lazy instances of 1 on the device
    lazy var floatOne = { DeviceValue<Float>(1) }()
    lazy var doubleOne = { DeviceValue<Double>(1) }()

    public init() {
        var handle: cublasHandle_t?
        !!cublasCreate_v2(&handle)
        self.handle = handle!

        /// We will use device pointer reference in this library
        /// for safety and type consistency
        pointerMode = .deviceReference
    }

    deinit {
        !!cublasDestroy_v2(handle)
    }

    open var version: Int {
        var version: Int32 = 0
        !!cublasGetVersion_v2(handle, &version)
        return Int(version)
    }

    open var stream: Stream? = nil {
        didSet {
            if let stream = stream {
                stream.withUnsafeHandle { streamHandle in
                    !!cublasSetStream_v2(handle, streamHandle)
                }
            } else {
                !!cublasSetStream_v2(handle, nil)
            }
        }
    }

    open var allowsAtomics: Bool {
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

    public enum PointerMode : UInt32 {
        case hostReference = 0
        case deviceReference = 1
    }

    open internal(set) var pointerMode: PointerMode {
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
