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

    public convenience init?(ptxFile path: String) {
        guard let ptx = try? PTX(contentsOfFile: path) else {
            return nil
        }
        self.init(ptx: ptx)
    }

    public init?(ptx: PTX) {
        var handle: CUmodule? = nil
        let result = ptx.data.withUnsafeBytes { bytes in
            cuModuleLoadData(&handle, bytes)
        }
        guard result == CUDA_SUCCESS else {
            return nil
        }
        self.handle = handle!
    }
    
}
