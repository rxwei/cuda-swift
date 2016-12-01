//
//  Error.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CNVRTC

public struct CompilerError : Error {

    public enum Kind : UInt32 {
        case outOfMemory = 1
        case programCreationFailure = 2
        case invalidInput = 3
        case invalidProgram = 4
        case invalidOption = 5
        case compilationError = 6
        case builtinOperationFailure = 7
        case noNameExpressionsAfterCompilation = 8
        case noLoweredNamesBeforeCompilation = 9
        case nameExpressionNotValid = 10
        case internalError = 11

        /// Invalid text encoding of input
        /// - Note: Not part of CUDA libraries
        case invalidEncoding

        init(_ nvrtcError: nvrtcResult) {
            self.init(rawValue: nvrtcError.rawValue)!
        }
    }

    public let kind: Kind
    public let log: String?

    public init(kind: Kind, log: String? = nil) {
        self.kind = kind
        self.log = log
    }

    init(result: nvrtcResult, log: String? = nil) {
        self.kind = Kind(result)
        self.log = log
    }

}

extension CompilerError : CustomStringConvertible {

    public var description: String {
        return log ?? String(describing: kind)
    }

}

func ensureSuccess(_ result: nvrtcResult) throws {
    guard result == NVRTC_SUCCESS else {
        throw CompilerError(kind: .init(result))
    }
}


func forceSuccess(_ result: nvrtcResult) {
    guard result == NVRTC_SUCCESS else {
        fatalError(String(describing: CompilerError(kind: .init(result))))
    }
}

prefix operator !!

@inline(__always)
prefix func !!(result: nvrtcResult) {
    forceSuccess(result)
}
