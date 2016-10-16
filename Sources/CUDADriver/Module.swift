//
//  File.swift
//  CUDA
//
//  Created by Richard Wei on 10/13/16.
//
//

import Foundation
import CCUDA

open class Module {

    let handle: CUmodule

    init(handle: CUmodule) {
        self.handle = handle
    }

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
    
}
