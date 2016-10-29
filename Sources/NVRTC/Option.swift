//
//  Option.swift
//  CUDA
//
//  Created by Richard Wei on 10/28/16.
//
//

@_exported import struct CUDADriver.ComputeCapability

public enum Option {
    case computeCapability(ComputeCapability)
    case relocatableDeviceCode(Bool)
    case debugInfo
    case lineInfo
    case maxRegisterCount(Int)
    case flushDenormalValues(Bool)
    case useFastMath
    case preciseSquareRoot(Bool)
    case preciseDivision(Bool)
    case contractIntoFMAD(Bool)
    case defineMacro(String, as: String?)
    case undefineMacro(String)
    case includePath(String)
    case preincludeHeader(String)
    case cpp11
    case builtinMoveAndForward(Bool)
    case builtinInitializerList(Bool)
    case disableWarnings
    case assertRestrictPointers
    case deviceAsDefaultExecutionSpace

    var rawArgument: String {
        switch self {
        case let .computeCapability(cc):
            return "--gpu-architecture=compute_\(cc.major)\(cc.minor)"
        case let .relocatableDeviceCode(r):
            return "--relocatable-device-code=\(r)"
        case .debugInfo:
            return "--device-debug"
        case .lineInfo:
            return "--generate-line-info"
        case let .maxRegisterCount(count):
            return "--maxrregcount=\(count)"
        case let .flushDenormalValues(ftz):
            return "--ftz=\(ftz)"
        case .useFastMath:
            return "-use_fast_math"
        case let .preciseSquareRoot(prec):
            return "--prec-sqrt=\(prec)"
        case let .preciseDivision(prec):
            return "--prec-div=\(prec)"
        case let .contractIntoFMAD(fmad):
            return "--fmad=\(fmad)"
        case let .defineMacro(macro, as: x):
            return x == nil ? "--define-macro=\(macro)"
                            : "--define-macro=\(macro)=\(x!)"
        case let .undefineMacro(macro):
            return "--undefine-macro=\(macro)"
        case let .includePath(path):
            return "--include-path=\(path)"
        case let .preincludeHeader(header):
            return "--pre-include=\(header)"
        case .cpp11:
            return "--std=c++11"
        case let .builtinMoveAndForward(builtin):
            return "--builtin-move-forward=\(builtin)"
        case let .builtinInitializerList(builtin):
            return "--builtin-initializer-list=\(builtin)"
        case .disableWarnings:
            return "--disable-warnings"
        case .assertRestrictPointers:
            return "--restrict"
        case .deviceAsDefaultExecutionSpace:
            return "--device-as-default-execution-space"
        }
    }

}
