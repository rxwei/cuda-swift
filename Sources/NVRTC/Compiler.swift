//
//  CUDACompiler.swift
//  SAD
//
//  Created by Richard Wei on 9/28/16.
//
//

import CNVRTC
@_exported import struct CUDADriver.PTX
import protocol CUDADriver.CHandleCarrier
import Foundation

/// Source-level program
open class Program {

    let handle: nvrtcProgram

    public var name: String?

    /// Initialize with source program and headers
    public init(data: Data, name: String? = nil,
                headers: [(name: String, data: Data)]) throws {
        self.name = name
        var headerNames: [UnsafePointer<Int8>?] = headers.map { name, _ in
            name.withCString{$0}
        }
        var headerCodes: [UnsafePointer<Int8>?] = headers.map { _, data in
            data.withUnsafeBytes{$0}
        }
        var handle: nvrtcProgram?
        try ensureSuccess(
            data.withUnsafeBytes { bytes in
                nvrtcCreateProgram(
                    &handle,                          /// Handle
                    bytes,                            /// Source
                    name?.utf8CString.map{$0} ?? nil, /// Program name
                    Int32(headerNames.count),         /// Header count
                    &headerCodes,                     /// Headers
                    &headerNames                      /// Header names
                )
            }
        )
        self.handle = handle! /// Safe
    }

    /// Initialize with source program
    public init(data: Data, name: String? = nil) throws {
        self.name = name
        var handle: nvrtcProgram?
        try ensureSuccess(
            data.withUnsafeBytes { bytes in
                nvrtcCreateProgram(
                    &handle,                          /// Handle
                    bytes,                            /// Source
                    name?.utf8CString.map{$0} ?? nil, /// Program name
                    0, nil, nil                       /// No headers
                )
            }
        )
        self.handle = handle! /// Safe
    }

    public convenience init(source: String, name: String? = nil) throws {
        guard let data = source.data(using: .utf8, allowLossyConversion: true) else {
            throw CompilerError.invalidEncoding
        }
        try self.init(data: data, name: name)
    }

    public convenience init(sourceFile: String) throws {
        let url = URL(fileURLWithPath: sourceFile)
        let name = url.deletingPathExtension().lastPathComponent
        let data = try Data(contentsOf: url)
        try self.init(data: data, name: name)
    }

    deinit {
        var handle: nvrtcProgram? = self.handle
        !!nvrtcDestroyProgram(&handle)
    }

    internal func retrievePTX() throws -> PTX {
        var ptxSize = 0
        /// Get PTX size
        try ensureSuccess(nvrtcGetPTXSize(handle, &ptxSize))
        var outData = Data(capacity: ptxSize)
        outData.count = ptxSize
        /// Get PTX data
        try outData.withUnsafeMutableBytes { ptxBuf in
            try ensureSuccess(nvrtcGetPTX(handle, ptxBuf))
        }
        return PTX(data: outData, name: name)
    }

}

open class Compiler {

    open class var version: (major: Int, minor: Int) {
        var major: Int32 = 0
        var minor: Int32 = 0
        nvrtcVersion(&major, &minor)
        return (major: Int(major), minor: Int(minor))
    }

    open class func compile(_ program: Program) throws -> PTX {
        try ensureSuccess(nvrtcCompileProgram(program.handle, 0, nil))
        return try program.retrievePTX()
    }

    open class func compile(_ program: Program, options: [String]) throws -> PTX {
        return try options.withUnsafeBufferPointer { ptr in
            var cStringPtr: [UnsafePointer<Int8>?] = ptr.map{$0.withCString{$0}}
            try ensureSuccess(
                nvrtcCompileProgram(program.handle, Int32(ptr.count), &cStringPtr)
            )
            return try program.retrievePTX()
        }
    }

}
