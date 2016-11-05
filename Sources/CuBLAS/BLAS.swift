//
//  BLAS.swift
//  CUDA
//
//  Created by Richard Wei on 9/27/16.
//
//

import CCuBLAS
import protocol CUDADriver.CHandleCarrier
@_exported import class CUDARuntime.Stream
@_exported import struct CUDARuntime.Device

open class BLAS : CHandleCarrier {

    private static var blasInstances: [Int : BLAS] = Dictionary(minimumCapacity: Device.count)

    open class func global(on device: Device = Device.current) -> BLAS {
        guard let blas = BLAS.blasInstances[device.index] else {
            let blas = BLAS()
            blasInstances[device.index] = blas
            return blas
        }
        return blas
    }

    public typealias Handle = cublasHandle_t

    let handle: cublasHandle_t

    public init() {
        var handle: cublasHandle_t?
        !!cublasCreate_v2(&handle)
        self.handle = handle!
        pointerMode = .hostReference
    }

    deinit {
        cublasDestroy_v2(handle)
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

    /// Whether the scalar values are passed by reference on the host or device
    public enum PointerMode : UInt32 {
        case hostReference = 0
        case deviceReference = 1
    }

    /// Whether the scalar values are passed by reference on the host or device.
    /// Currently only host reference is supported.
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

    public func withUnsafeHandle<Result>(_ body: (Handle) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

}
