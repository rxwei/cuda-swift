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
    }

    public var version: Int {
        var version: Int32 = 0
        cublasGetVersion_v2(handle, &version)
        return Int(version)
    }

    public var stream: Stream? {
        var streamHandle: cudaStream_t?
        !!cublasGetStream_v2(handle, &streamHandle)
        return streamHandle.flatMap(Stream.current(with:))
    }

    deinit {
        cublasDestroy_v2(handle)
    }

}
