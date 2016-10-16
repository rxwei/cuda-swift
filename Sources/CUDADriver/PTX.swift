//
//  PTX.swift
//  CUDA
//
//  Created by Richard Wei on 10/13/16.
//
//

import Foundation

public struct PTX {

    let data: Data
    public var name: String?

    public init(data: Data, name: String? = nil) {
        self.data = data
        self.name = name
    }

    public init(contentsOfFile path: String) throws {
        try self.init(contentsOf: URL(fileURLWithPath: path))
    }

    public init(contentsOf url: URL) throws {
        data = try Data(contentsOf: url)
        name = url.deletingPathExtension().lastPathComponent
    }

    public func write(to url: URL) throws {
        try data.write(to: url)
    }

    public func write(toFile path: String) throws {
        try write(to: URL(fileURLWithPath: path))
    }
    
}
