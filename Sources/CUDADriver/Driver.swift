//
//  Driver.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

import CCUDA

public enum Driver {

    /// Initialize driver API
    /// Call this before using any driver API!
    public static func initialize() throws {
        try ensureSuccess(cuInit(0))
    }

}
