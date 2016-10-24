//
//  File.swift
//  CUDA
//
//  Created by Richard Wei on 10/13/16.
//
//

import Foundation
import CCUDA

open class Module : CHandleCarrier {

    public typealias Handle = CUmodule

    let handle: CUmodule

    deinit {
        cuModuleUnload(handle)
    }

    public convenience init(ptxFile path: String) throws {
        try self.init(ptx: PTX(contentsOfFile: path))
    }

    public init(ptx: PTX) throws {
        var handle: CUmodule? = nil
        try ptx.data.withUnsafeBytes { bytes in
            try ensureSuccess(cuModuleLoadData(&handle, bytes))
        }
        self.handle = handle!
    }

    open func withUnsafeHandle<Result>
        (_ body: (CUmodule) throws -> Result) rethrows -> Result {
        return try body(handle)
    }

    open func function(named name: String) -> Function? {
        var function: CUfunction?
        do {
            try name.withCString { cStr in
                try ensureSuccess(cuModuleGetFunction(&function, handle, cStr))
            }
        } catch {
            return nil
        }
        return Function(function!)
    }

}
