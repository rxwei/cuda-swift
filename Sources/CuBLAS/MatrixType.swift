//
//  MatrixType.swift
//  CUDA
//
//  Created by Richard Wei on 10/25/16.
//
//

import CCuBLAS
@_exported import struct CUDARuntime.DeviceValue

public protocol RowMajorOrderedMatrix : RandomAccessCollection {

    associatedtype Element
    associatedtype Cell
    associatedtype Row : Collection

    init(rowCount: Int, columnCount: Int, rowMajorElements: [Element])

    subscript(i: Int) -> Row { get set }

    subscript(i: Int, j: Int) -> Cell { get set }

}

public extension RowMajorOrderedMatrix {

    public typealias Index = Int
    public typealias IndexDistance = Int

    public func index(after i: Int) -> Int {
        return i + 1
    }

    public func index(before i: Int) -> Int {
        return i - 1
    }

    public var startIndex: Int {
        return 0
    }

    public var endIndex: Int {
        return count
    }
    
}

public struct DeviceMatrix<Element> : RowMajorOrderedMatrix {
    
    var linearStorage: DeviceVector<Element>

    public let rowCount: Int, columnCount: Int

    public var count: Int {
        return rowCount
    }

    public init(rowCount: Int, columnCount: Int, rowMajorElements: [Element]) {
        self.rowCount = rowCount
        self.columnCount = columnCount
        linearStorage = DeviceVector(capacity: rowCount * columnCount)
        linearStorage.withUnsafeMutableDevicePointer { devPtr in
            rowMajorElements.withUnsafeBufferPointer { buf in
                !!cublasSetMatrix(
                    Int32(rowCount),
                    Int32(columnCount),
                    Int32(MemoryLayout<Element>.stride),
                    buf.baseAddress,
                    Int32(rowCount),
                    devPtr.deviceAddress,
                    Int32(rowCount)
                )
            }
        }
    }
    
    public subscript(i: Int) -> DeviceVector<Element> {
        get {
            let base = i * columnCount
            return linearStorage[base..<(base + columnCount)]
        }
        set {
            let base = i * columnCount
            linearStorage[base..<(base + columnCount)] = newValue
        }
    }

    public subscript(i: Int, j: Int) -> DeviceValue<Element> {
        get {
            return linearStorage[i * columnCount + j]
        }
        set {
            linearStorage[i * columnCount + j] = newValue
        }
    }

}
