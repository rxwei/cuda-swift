//
//  CUDACompiler.swift
//  SAD
//
//  Created by Richard Wei on 9/28/16.
//
//

import CNVRTC
@_exported import struct CUDADriver.PTX
import Foundation

open class Compiler {

    open class var version: (major: Int, minor: Int) {
        var major: Int32 = 0
        var minor: Int32 = 0
        nvrtcVersion(&major, &minor)
        return (major: Int(major), minor: Int(minor))
    }
    
    open class func compile(_ data: Data,
                            named name: String? = nil,
                            options: [String]? = nil) throws -> PTX {
        /// Create program
        var program: nvrtcProgram?
        try ensureSuccess(
            data.withUnsafeBytes { (bytes: UnsafePointer<Int8>) in
                if var nameStr = name?.utf8CString.map({$0}) {
                    return nvrtcCreateProgram(&program, bytes, &nameStr, 0, nil, nil)
                }
                return nvrtcCreateProgram(&program, bytes, "default", 0, nil, nil)
            }
        )

        /// Compile with options
        if let options = options {
            try options.withUnsafeBufferPointer { (ptr: UnsafeBufferPointer<String>) in
                var cStringPtr: [UnsafePointer<Int8>?] = ptr.map{$0.withCString{$0}}
                try ensureSuccess(nvrtcCompileProgram(program, Int32(ptr.count), &cStringPtr))
            }
        }
        /// Compile without options
        else {
            try ensureSuccess(nvrtcCompileProgram(program, 0, nil))
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
    
    open class func compile(from url: URL,
                            options: [String]? = nil) throws -> PTX {
        let handle = try FileHandle(forReadingFrom: url)
        let data = handle.readDataToEndOfFile()
        return try compile(
            data,
            named: url.deletingPathExtension().lastPathComponent,
            options: options
        )
    }
    
    open class func compile(_ source: String,
                            named name: String? = nil,
                            options: [String]? = nil) throws -> PTX {
        guard let data = source.data(using: .utf8, allowLossyConversion: true) else {
            throw CompilerError.wrongSourceFormat
        }
        return try compile(data, named: name, options: options)
    }
    
    open class func compileSourceFile(_ path: String,
                                      options: [String]? = nil) throws -> PTX {
        return try compile(from: URL(fileURLWithPath: path), options: options)
    }

}
