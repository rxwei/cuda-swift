//
//  CHandleCarrier.swift
//  CUDA
//
//  Created by Richard Wei on 10/23/16.
//
//

public protocol CHandleCarrier {
    associatedtype Handle
    func withUnsafeHandle<Result>(_ body: (Handle) throws -> Result) rethrows -> Result
}

