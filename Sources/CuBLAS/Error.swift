//
//  Error.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

import CCuBLAS

public enum BLASError : UInt32, Error {
    case notInitialized       = 1
    case allocationFailed     = 3
    case invalidValue         = 7
    case architectureMismatch = 8
    case mappingError         = 11
    case executionFailed      = 13
    case internalError        = 14
    case notSupported         = 15
    case licenseError         = 16

    init(_ cudaError: cublasStatus_t) {
        self.init(rawValue: cudaError.rawValue)!
    }
}

func ensureSuccess(_ result: cublasStatus_t) throws {
    guard result == CUBLAS_STATUS_SUCCESS else {
        throw BLASError(result)
    }
}

prefix operator !!

func forceSuccess(_ result: cublasStatus_t) {
    guard result == CUBLAS_STATUS_SUCCESS else {
        fatalError(String(describing: BLASError(result)))
    }
}

@inline(__always)
prefix func !!(status: cublasStatus_t) {
    forceSuccess(status)
}
