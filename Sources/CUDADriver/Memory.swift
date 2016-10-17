//
//  Memory.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public enum SharedMemoryBankSize: UInt32 {
    case `default` = 0x00
    case fourBytes = 0x01
    case eightBytes = 0x02
}

/// TODO: DevicePointer
/// This needs a LOT of work!

/*

public struct UnsafeMutableDevicePointer<Pointee> {

    public typealias Distance = Int
    
    /// Converts from an opaque pointer to a typed pointer.
    public init(_ from: OpaquePointer)
    
    /// Converts from an opaque pointer to a typed pointer.
    ///
    /// Returns `nil` if `from` is `nil`.
    public init?(_ from: OpaquePointer?)
    
    /// Creates an `UnsafeMutablePointer` with a given pattern of bits.
    ///
    /// Returns `nil` if `bitPattern` is zero.
    public init?(bitPattern: Int)
    
    /// Creates an `UnsafeMutablePointer` with a given pattern of bits.
    ///
    /// Returns `nil` if `bitPattern` is zero.
    public init?(bitPattern: UInt)
    
    /// Creates an `UnsafeMutablePointer` from another `UnsafeMutablePointer`.
    public init(_ other: UnsafeMutableDevicePointer<Pointee>)
    
    /// Creates an `UnsafeMutablePointer` from another `UnsafeMutablePointer`.
    ///
    /// Returns `nil` if `other` is `nil`.
    public init?(_ other: UnsafeMutableDevicePointer<Pointee>?)
    
    /// Converts from `UnsafePointer` to `UnsafeMutablePointer` of the same
    /// `Pointee`.
    public init(mutating other: UnsafeDevicePointer<Pointee>)
    
    /// Converts from `UnsafePointer` to `UnsafeMutablePointer` of the same
    /// `Pointee`.
    ///
    /// Returns nil if `bitPattern` is zero.
    public init?(mutating other: UnsafeDevicePointer<Pointee>?)
    
    /// Allocates and points at uninitialized aligned memory for `count`
    /// instances of `Pointee`.
    ///
    /// - Postcondition: The pointee is allocated, but not initialized.
    public static func allocate(capacity count: Int) -> UnsafeMutableDevicePointer<Pointee>
    
    /// Deallocates uninitialized memory allocated for `count` instances
    /// of `Pointee`.
    ///
    /// - Precondition: The memory is not initialized.
    ///
    /// - Postcondition: The memory has been deallocated.
    public func deallocate(capacity: Int)
    
    /// Accesses the `Pointee` instance referenced by `self`.
    ///
    /// - Precondition: the pointee has been initialized with an instance of
    ///   type `Pointee`.
    public var pointee: Pointee { get nonmutating set }
    
    /// Initializes `self.pointee` with `count` consecutive copies of `newValue`
    ///
    /// - Precondition: The pointee is not initialized.
    ///
    /// - Precondition: `count` is non-negative.
    ///
    /// - Postcondition: The pointee is initialized; the value should eventually
    ///   be destroyed or moved from to avoid leaks.
    public func initialize(to newValue: Pointee, count: Int = default)
    
    /// Retrieves the `pointee`, returning the referenced memory to an
    /// uninitialized state.
    ///
    /// Equivalent to `{ defer { deinitialize() }; return pointee }()`, but
    /// more efficient.
    ///
    /// - Precondition: The pointee is initialized.
    ///
    /// - Postcondition: The memory is uninitialized.
    public func move() -> Pointee
    
    /// Replaces `count` initialized `Pointee`s starting at `self` with
    /// the `count` `Pointee`s at `source`.
    ///
    /// - Precondition: `count >= 0`
    ///
    /// - Precondition: The `Pointee`s at `self..<self + count` and
    ///   `source..<source + count` are initialized.
    public func assign(from source: UnsafeDevicePointer<Pointee>, count: Int)
    
    /// Initializes memory starting at `self` with `count` `Pointee`s
    /// beginning at `source`, and returning the source memory to an
    /// uninitialized state.
    ///
    /// - Precondition: `count >= 0`
    ///
    /// - Precondition: The memory at `self..<self + count` is uninitialized
    ///   and the `Pointees` at `source..<source + count` are
    ///   initialized.
    ///
    /// - Postcondition: The `Pointee`s at `self..<self + count` are
    ///   initialized and the memory at `source..<source + count` is
    ///   uninitialized.
    public func moveInitialize(from source: UnsafeMutableDevicePointer<Pointee>, count: Int)
    
    /// Initializes memory starting at `self` with `count` `Pointee`s
    /// beginning at `source`.
    ///
    /// - Precondition: `count >= 0`
    ///
    /// - Precondition: The memory regions `source..<source + count`
    ///   and `self..<self + count` do not overlap.
    ///
    /// - Precondition: The memory at `self..<self + count` is uninitialized
    ///   and the `Pointees` at `source..<source + count` are
    ///   initialized.
    ///
    /// - Postcondition: The `Pointee`s at `self..<self + count` and
    ///   `source..<source + count` are initialized.
    public func initialize(from source: UnsafeDevicePointer<Pointee>, count: Int)
    
    /// Initializes memory starting at `self` with the elements of `source`.
    ///
    /// - Precondition: The memory at `self..<self + count` is
    ///   uninitialized.
    ///
    /// - Postcondition: The `Pointee`s at `self..<self + count` are
    ///   initialized.
    public func initialize<C : Collection where C.Iterator.Element == Pointee>(from source: C)
    
    /// Replaces `count` initialized `Pointee`s starting at `self` with
    /// the `count` `Pointee`s starting at `source`, returning the
    /// source memory to an uninitialized state.
    ///
    /// - Precondition: `count >= 0`
    ///
    /// - Precondition: The memory regions `source..<source + count`
    ///   and `self..<self + count` do not overlap.
    ///
    /// - Precondition: The `Pointee`s at `self..<self + count` and
    ///   `source..<source + count` are initialized.
    ///
    /// - Postcondition: The `Pointee`s at `self..<self + count` are
    ///   initialized and the `Pointees` at `source..<source + count`
    ///   are uninitialized.
    public func moveAssign(from source: UnsafeMutableDevicePointer<Pointee>, count: Int)
    
    /// De-initializes the `count` `Pointee`s starting at `self`, returning
    /// their memory to an uninitialized state.
    ///
    /// Returns an UnsafeMutableRawPointer to this memory.
    ///
    /// - Precondition: The `Pointee`s at `self..<self + count` are
    ///   initialized.
    ///
    /// - Postcondition: The memory is uninitialized.
    public func deinitialize(count: Int = default) -> UnsafeMutableRawPointer
    
    /// Rebinds memory at `self` to type `T` with capacity to hold `count`
    /// adjacent `T` values while executing the `body` closure. After executing
    /// the closure, rebinds memory back to `Pointee`.
    ///
    /// - Precondition: Type 'T' is layout compatible with type 'Pointee'.
    /// - Precondition: The memory `self..<self + count * MemoryLayout<T>.stride`
    ///   is bound to `Pointee`.
    public func withMemoryRebound<T, Result>(to: T.Type, capacity count: Int, _ body: (UnsafeMutableDevicePointer<T>) throws -> Result) rethrows -> Result
    
    /// Accesses the pointee at `self + i`.
    ///
    /// - Precondition: the pointee at `self + i` is initialized.
    public subscript(i: Int) -> Pointee { get nonmutating set }
    
    /// The pointer's hash value.
    ///
    /// The hash value is not guaranteed to be stable across different
    /// invocations of the same program.  Do not persist the hash value across
    /// program runs.
    public var hashValue: Int { get }
    
    /// Returns the next consecutive position.
    public func successor() -> UnsafeMutableDevicePointer<Pointee>
    
    /// Returns the previous consecutive position.
    public func predecessor() -> UnsafeMutableDevicePointer<Pointee>
    
    /// Returns `end - self`.
    public func distance(to x: UnsafeMutableDevicePointer<Pointee>) -> Int
    
    /// Returns `self + n`.
    public func advanced(by n: Int) -> UnsafeMutableDevicePointer<Pointee>
}

extension UnsafeMutableDevicePointer : CustomDebugStringConvertible {

    /// A textual representation of the pointer, suitable for debugging.
    public var debugDescription: String { get }
}

extension UnsafeMutableDevicePointer : CustomReflectable {

    /// The custom mirror for this instance.
    ///
    /// If this type has value semantics, the mirror should be unaffected by
    /// subsequent mutations of the instance.
    public var customMirror: Mirror { get }
}

extension UnsafeMutableDevicePointer : CustomPlaygroundQuickLookable {

    /// A custom playground Quick Look for this instance.
    ///
    /// If this type has value semantics, the `PlaygroundQuickLook` instance
    /// should be unaffected by subsequent mutations.
    public var customPlaygroundQuickLook: PlaygroundQuickLook { get }
}

extension UnsafeMutableDevicePointer : CVarArg {
}

/// - Note: Strideable's implementation is potentially less efficient and cannot
///   handle misaligned pointers.
public func ==<Pointee>(lhs: UnsafeMutableDevicePointer<Pointee>, rhs: UnsafeMutableDevicePointer<Pointee>) -> Bool

/// - Note: Strideable's implementation is potentially less efficient and cannot
///   handle misaligned pointers.
///
/// - Note: This is an unsigned comparison unlike Strideable's implementation.
public func <<Pointee>(lhs: UnsafeMutableDevicePointer<Pointee>, rhs: UnsafeMutableDevicePointer<Pointee>) -> Bool

/// - Note: The following family of operator overloads are redundant
///   with Strideable. However, optimizer improvements are needed
///   before they can be removed without affecting performance.
public func +<Pointee>(lhs: UnsafeMutableDevicePointer<Pointee>, rhs: Int) -> UnsafeMutableDevicePointer<Pointee>

public func +<Pointee>(lhs: Int, rhs: UnsafeMutableDevicePointer<Pointee>) -> UnsafeMutableDevicePointer<Pointee>

public func -<Pointee>(lhs: UnsafeMutableDevicePointer<Pointee>, rhs: Int) -> UnsafeMutableDevicePointer<Pointee>

public func -<Pointee>(lhs: UnsafeMutableDevicePointer<Pointee>, rhs: UnsafeMutableDevicePointer<Pointee>) -> Int

public func +=<Pointee>(lhs: inout UnsafeMutableDevicePointer<Pointee>, rhs: Int)

public func -=<Pointee>(lhs: inout UnsafeMutableDevicePointer<Pointee>, rhs: Int)

/// A raw pointer for accessing data of type `Pointee`.  This type
/// provides no automated memory management, and therefore must
/// be handled with great care to ensure safety.
///
/// Instances must be aligned to `MemoryLayout<Pointee>.alignment`, i.e.
/// `(UnsafePointer<Int8>(self) - nil) % MemoryLayout<Pointee>.alignment == 0`
///
/// The memory referenced by an instance can be in one of the following states:
///
/// - Memory is not allocated (for example, pointer is null, or memory has
///   been deallocated previously).
///
/// - Memory is allocated, but value has not been initialized.
///
/// - Memory is allocated and value is initialized.
public struct UnsafeDevicePointer<Pointee> : Strideable, Hashable {

    public typealias Distance = Int

    /// Converts from an opaque pointer to a typed pointer.
    public init(_ from: OpaquePointer)

    /// Converts from an opaque pointer to a typed pointer.
    ///
    /// Returns `nil` if `from` is `nil`.
    public init?(_ from: OpaquePointer?)

    /// Creates an `UnsafePointer` with a given pattern of bits.
    ///
    /// Returns `nil` if `bitPattern` is zero.
    public init?(bitPattern: Int)

    /// Creates an `UnsafePointer` with a given pattern of bits.
    ///
    /// Returns `nil` if `bitPattern` is zero.
    public init?(bitPattern: UInt)

    /// Creates an `UnsafePointer` from another `UnsafePointer`.
    public init(_ other: UnsafeDevicePointer<Pointee>)

    /// Creates an `UnsafePointer` from another `UnsafePointer`.
    ///
    /// Returns `nil` if `other` is `nil`.
    public init?(_ other: UnsafeDevicePointer<Pointee>?)

    /// Converts from `UnsafeMutablePointer` to an `UnsafePointer` of the same `Pointee`.
    public init(_ other: UnsafeMutableDevicePointer<Pointee>)

    /// Converts from `UnsafeMutablePointer` to an `UnsafePointer` of the same `Pointee`.
    ///
    /// Returns nil if `from` is nil.
    public init?(_ other: UnsafeMutableDevicePointer<Pointee>?)

    /// Accesses the `Pointee` instance referenced by `self`.
    ///
    /// - Precondition: the pointee has been initialized with an instance of
    ///   type `Pointee`.
    public var pointee: Pointee { get }

    /// Rebinds memory at `self` to type `T` with capacity to hold `count`
    /// adjacent `T` values while executing the `body` closure. After executing
    /// the closure, rebinds memory back to `Pointee`.
    ///
    /// - Precondition: Type 'T' is layout compatible with type 'Pointee'.
    /// - Precondition: The memory `self..<self + count * MemoryLayout<T>.stride`
    ///   is bound to `Pointee`.
    public func withMemoryRebound<T, Result>(to: T.Type, capacity count: Int, _ body: (UnsafeDevicePointer<T>) throws -> Result) rethrows -> Result

    /// Accesses the pointee at `self + i`.
    ///
    /// - Precondition: the pointee at `self + i` is initialized.
    public subscript(i: Int) -> Pointee { get }

    /// The pointer's hash value.
    ///
    /// The hash value is not guaranteed to be stable across different
    /// invocations of the same program.  Do not persist the hash value across
    /// program runs.
    public var hashValue: Int { get }

    /// Returns the next consecutive position.
    public func successor() -> UnsafeDevicePointer<Pointee>

    /// Returns the previous consecutive position.
    public func predecessor() -> UnsafeDevicePointer<Pointee>

    /// Returns `end - self`.
    public func distance(to x: UnsafeDevicePointer<Pointee>) -> Int

    /// Returns `self + n`.
    public func advanced(by n: Int) -> UnsafeDevicePointer<Pointee>
}

extension UnsafeDevicePointer : CustomDebugStringConvertible {

    /// A textual representation of the pointer, suitable for debugging.
    public var debugDescription: String { get }
}

extension UnsafeDevicePointer : CustomReflectable {

    /// The custom mirror for this instance.
    ///
    /// If this type has value semantics, the mirror should be unaffected by
    /// subsequent mutations of the instance.
    public var customMirror: Mirror { get }
}

extension UnsafeDevicePointer : CustomPlaygroundQuickLookable {

    /// A custom playground Quick Look for this instance.
    ///
    /// If this type has value semantics, the `PlaygroundQuickLook` instance
    /// should be unaffected by subsequent mutations.
    public var customPlaygroundQuickLook: PlaygroundQuickLook { get }
}

extension UnsafeDevicePointer : CVarArg {
}

/// - Note: Strideable's implementation is potentially less efficient and cannot
///   handle misaligned pointers.
public func ==<Pointee>(lhs: UnsafeDevicePointer<Pointee>, rhs: UnsafeDevicePointer<Pointee>) -> Bool

/// - Note: Strideable's implementation is potentially less efficient and cannot
///   handle misaligned pointers.
///
/// - Note: This is an unsigned comparison unlike Strideable's implementation.
public func <<Pointee>(lhs: UnsafeDevicePointer<Pointee>, rhs: UnsafeDevicePointer<Pointee>) -> Bool

/// - Note: The following family of operator overloads are redundant
///   with Strideable. However, optimizer improvements are needed
///   before they can be removed without affecting performance.
public func +<Pointee>(lhs: UnsafeDevicePointer<Pointee>, rhs: Int) -> UnsafeDevicePointer<Pointee>

public func +<Pointee>(lhs: Int, rhs: UnsafeDevicePointer<Pointee>) -> UnsafeDevicePointer<Pointee>

public func -<Pointee>(lhs: UnsafeDevicePointer<Pointee>, rhs: Int) -> UnsafeDevicePointer<Pointee>

public func -<Pointee>(lhs: UnsafeDevicePointer<Pointee>, rhs: UnsafeDevicePointer<Pointee>) -> Int

public func +=<Pointee>(lhs: inout UnsafeDevicePointer<Pointee>, rhs: Int)

public func -=<Pointee>(lhs: inout UnsafeDevicePointer<Pointee>, rhs: Int)
 
 */
