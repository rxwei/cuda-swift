//
//  Array.swift
//  Warp
//
//  Created by Richard Wei on 10/19/16.
//
//

import CUDARuntime

protocol DeviceArrayProtocol :
    DeviceAddressable, MutableDeviceCollection, ExpressibleByArrayLiteral
{
    associatedtype Buffer : AnyObject
    var capacity: Int { get }
    subscript(i: Int) -> Iterator.Element { get set }
    var buffer: Buffer { get set }
    init()
    init(device: Device)
    init(_ buffer: Buffer)
}

/// Array value type with a contiguous storage on CUDA device
public struct DeviceArray<Element> : DeviceCollection, DeviceArrayProtocol {

    public typealias Index = Int
    public typealias IndexDistance = Int
    public typealias SubSequence = DeviceArray<Element>

    var buffer: DeviceArrayBuffer<Element>

    /// Copy on write
    private var mutatingBuffer: DeviceArrayBuffer<Element> {
        mutating get {
            if !isKnownUniquelyReferenced(&buffer) {
                buffer = DeviceArrayBuffer(buffer)
            }
            return buffer
        }
    }

    init(_ buffer: DeviceArrayBuffer<Element>) {
        self.buffer = buffer
    }

    public var device: Device {
        return buffer.device
    }

    /// Create an array on the current CUDA device.
    public init() {
        buffer = DeviceArrayBuffer(device: Device.current, capacity: 0)
    }

    /// Create an array on a specific CUDA device
    ///
    /// - Parameter device: CUDA device
    public init(device: Device) {
        buffer = DeviceArrayBuffer(device: device, capacity: 0)
    }

    /// Create an array of specified capacity on specified CUDA device
    ///
    /// - Parameters:
    ///   - device: CUDA device
    ///   - capacity: capacity of the array
    public init(device: Device, capacity: Int) {
        buffer = DeviceArrayBuffer(device: device, capacity: capacity)
    }

    /// Create an array of specified capacity on the current CUDA device
    ///
    /// - Parameters:
    ///   - capacity: capacity of the array
    public init(capacity: Int) {
        buffer = DeviceArrayBuffer(device: Device.current, capacity: capacity)
    }

    public init(repeating element: Element, count: Int) {
        buffer = DeviceArrayBuffer(repeating: element, count: count, device: Device.current)
    }

    public init(repeating element: Element, count: Int, device: Device) {
        buffer = DeviceArrayBuffer(repeating: element, count: count, device: device)
    }

    public init<C: Collection>(_ elements: C) where
        C.Iterator.Element == Element, C.IndexDistance == Int
    {
        buffer = DeviceArrayBuffer(elements, device: Device.current)
    }

    public init<C: Collection>(_ elements: C, device: Device) where
        C.Iterator.Element == Element, C.IndexDistance == Int
    {
        buffer = DeviceArrayBuffer(elements, device: device)
    }

    public init(arrayLiteral elements: Element...) {
        buffer = DeviceArrayBuffer(elements, device: Device.current)
    }

    /// Create an array by copying from the other array
    ///
    /// - Parameter other: array to assign to self
    public init(_ other: DeviceArray<Element>) {
        buffer = DeviceArrayBuffer(other.buffer)
    }

    private init(viewing buffer: DeviceArrayBuffer<Element>, range: Range<Int>) {
        self.buffer = DeviceArrayBuffer(viewing: buffer, in: range)
    }

    public var hostArray: [Element] {
        var elements: [Element] = []
        elements.reserveCapacity(count)
        /// Temporary array copy solution
        var temp = UnsafeMutablePointer<Element>.allocate(capacity: count)
        temp.assign(fromDevice: buffer.startAddress, count: count)
        elements.append(contentsOf: UnsafeBufferPointer(start: temp, count: count))
        temp.deallocate(capacity: count)
        return elements
    }

    /// - note: Currently `DeviceArray` is length-immutable. Need to implement variable
    /// startIndex/endIndex to support `.reserveCapacity(_:)` and `.append(_:)`. In that
    /// case, we'll be conforming to `RangeReplaceableCollection`
    var capacity: Int {
        return buffer.count
    }

    public var count: Int {
        return buffer.count
    }

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
        return buffer.count
    }

    public var indices: CountableRange<Int> {
        return 0..<count
    }

    public mutating func replaceSubrange
        (_ subrange: Range<Int>, with newElements: DeviceArray<Element>) {
        mutatingBuffer.replaceSubrange(bufferRange(fromLocal: subrange), with: newElements.buffer)
    }

    public mutating func replaceSubrange<C : Collection>
        (_ subrange: Range<Int>, with newElements: C) where C.Iterator.Element == DeviceValue<Element> {
        buffer.replaceSubrange(bufferRange(fromLocal: subrange), with: newElements.map{$0.buffer})
    }

    public subscript(i: Int) -> DeviceValue<Element> {
        get {
            return DeviceValue(buffer: DeviceValueBuffer(viewing: buffer, offsetBy: i))
        }
        set {
            var newValue = newValue
            mutatingBuffer[bufferIndex(fromLocal: i)] = newValue.mutatingBuffer
        }
    }

    @inline(__always)
    private func bufferRange(fromLocal localRange: Range<Int>) -> Range<Int> {
        return (buffer.startIndex + localRange.lowerBound)..<(buffer.startIndex + localRange.upperBound)
    }

    @inline(__always)
    private func bufferIndex(fromLocal localIndex: Int) -> Int {
        return localIndex + buffer.startIndex
    }

    public subscript(range: Range<Int>) -> DeviceArray<Element> {
        get {
            return DeviceArray(viewing: buffer, range: bufferRange(fromLocal: range))
        }
        mutating set {
            replaceSubrange(range, with: newValue)
        }
    }

    /// Calls a closure with a mutable pointer to the array's contiguous storage.
    ///
    /// - Parameter body: body closure with a pointer made available
    /// - Returns: whatever the closure returns
    /// - Throws: whatever the closure throws
    /// - Note: Do not return the pointer from the closure
    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (inout UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result {
        var startAddress = mutatingBuffer.startAddress
        return try body(&startAddress)
    }

    /// Calls a closure with a pointer to the array's contiguous storage.
    ///
    /// - Parameter body: body closure with a pointer made available
    /// - Returns: whatever the closure returns
    /// - Throws: whatever the closure throws
    /// - Note: Do not return the pointer from the closure
    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try body(UnsafeDevicePointer(buffer.startAddress))
    }

    /// Get the mutable device pointer to the start of the array.
    /// - Note: This should **only** be used when the lifetime of the pointer is
    /// shorter than the lifetime of the array. If you are not sure about the life
    /// time of the array, use `.withUnsafeMutableDevicePointer(...)` instead!
    public var unsafeMutableDevicePointer: UnsafeMutableDevicePointer<Element> {
        mutating get {
            return mutatingBuffer.startAddress
        }
    }

    /// Get the device pointer to the start of the array.
    /// - Note: This should **only** be used when the lifetime of the pointer is
    /// shorter than the lifetime of the array. If you are not sure about the life
    /// time of the array, use `.withUnsafeMutableDevicePointer(...)` instead!
    public var unsafeDevicePointer: UnsafeDevicePointer<Element> {
        return UnsafeDevicePointer(buffer.startAddress)
    }

}

public extension Array {

    /// Initialize an array by copying an array from CUDA device
    ///
    /// - Parameter deviceElements: array on CUDA device
    public init(_ deviceElements: DeviceArray<Element>) {
        self = deviceElements.hostArray
    }

}
