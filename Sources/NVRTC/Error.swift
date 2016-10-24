//
//  Error.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CNVRTC

public enum CompilerError : UInt32, Error {

    case outOfMemory = 1
    case programCreationFailure = 2
    case invalidInput = 3
    case invalidProgram = 4
    case invalidOption = 5
    case compilation = 6
    case builtinOperationFailure = 7
    case noNameExpressionsAfterCompilation = 8
    case noLoweredNamesBeforeCompilation = 9
    case nameExpressionNotValid = 10
    case internalError = 11
    case wrongSourceFormat

    init(_ nvrtcError: nvrtcResult) {
        self.init(rawValue: nvrtcError.rawValue)!
    }

}

func ensureSuccess(_ result: nvrtcResult) throws {
    guard result == NVRTC_SUCCESS else {
        throw CompilerError(result)
    }
}


func forceSuccess(_ result: nvrtcResult) {
    guard result == NVRTC_SUCCESS else {
        fatalError(String(describing: CompilerError(result)))
    }
}

prefix operator !!

@inline(__always)
prefix func !!(result: nvrtcResult) {
    forceSuccess(result)
}
