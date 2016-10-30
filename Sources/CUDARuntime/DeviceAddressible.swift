//
//  DeviceAddressible.swift
//  CUDA
//
//  Created by Richard Wei on 10/29/16.
//
//

public protocol DeviceAddressible {
    associatedtype Element
    var unsafePointer: UnsafeDevicePointer<Element> { get }
    func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result
    mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result
}
