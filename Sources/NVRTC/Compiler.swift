//
//  CUDACompiler.swift
//  SAD
//
//  Created by Richard Wei on 9/28/16.
//
//

import CNVRTC
import CUDADriver
import Foundation

public enum CUDACompiler {

    public enum Error : Swift.Error {
        case wrongSourceFormat
        case cannotCreateProgram
        case compilationError
    }

    static func compileSource(_ data: Data,
                              named name: String? = nil,
                              options: [String]? = nil) throws -> PTX {
        /// Create program
        var program: nvrtcProgram?
        let result = data.withUnsafeBytes { (bytes: UnsafePointer<Int8>) -> (nvrtcResult) in
            if var nameStr = name?.utf8CString.map({$0}) {
                return nvrtcCreateProgram(&program, bytes, &nameStr, 0, nil, nil)
            }
            return nvrtcCreateProgram(&program, bytes, "default", 0, nil, nil)
        }
        guard result == NVRTC_SUCCESS else {
            throw Error.cannotCreateProgram
        }

        /// Compile with options
        if let options = options {
            try options.withUnsafeBufferPointer { (ptr: UnsafeBufferPointer<String>) in
                var cStringPtr: [UnsafePointer<Int8>?] = ptr.map{$0.withCString{$0}}
                guard nvrtcCompileProgram(program, Int32(ptr.count), &cStringPtr) == NVRTC_SUCCESS else {
                    throw Error.compilationError
                }
            }
        }
        /// Compile without options
        else {
            guard nvrtcCompileProgram(program, 0, nil) == NVRTC_SUCCESS else {
                throw Error.compilationError
            }
        }

        /// Get PTX size
        var ptxSize: Int = 0
        nvrtcGetPTXSize(program, &ptxSize)

        /// Get PTX
        var outData = Data(capacity: ptxSize)
        outData.count = ptxSize
        outData.withUnsafeMutableBytes { ptxBuf -> () in
            nvrtcGetPTX(program, ptxBuf)
        }
        nvrtcDestroyProgram(&program)
        return PTX(data: outData, name: name)
    }

    static func compileSource(from url: URL,
                              options: [String]? = nil) throws -> PTX {
        let handle = try FileHandle(forReadingFrom: url)
        let data = handle.readDataToEndOfFile()
        return try compileSource(data,
                                 named: url.deletingPathExtension().lastPathComponent,
                                 options: options)
    }

    static func compileSource(_ source: String,
                              named name: String? = nil,
                              options: [String]? = nil) throws -> PTX {
        guard let data = source.data(using: .utf8, allowLossyConversion: true) else {
            throw Error.wrongSourceFormat
        }
        return try compileSource(data, named: name, options: options)
    }

    static func compileSourceFile(_ path: String,
                                  options: [String]? = nil) throws -> PTX {
        return try compileSource(from: URL(fileURLWithPath: path), options: options)
    }

}

public extension PTX {

    public init(compilingSourceFile path: String,
                options: [String]? = nil) throws {
        self = try CUDACompiler.compileSourceFile(path, options: options)
    }

    public init(compilingSourceFrom url: URL,
                options: [String]? = nil) throws {
        self = try CUDACompiler.compileSource(from: url, options: options)
    }

    public init(compilingSource source: Data,
                named name: String? = nil,
                options: [String]? = nil) throws {
        self = try CUDACompiler.compileSource(source, named: name, options: options)
    }

    public init(compilingSource source: String,
                named name: String? = nil,
                options: [String]? = nil) throws {
        self = try CUDACompiler.compileSource(source, named: name, options: options)
    }

}
