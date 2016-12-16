//
//  KernelSources.swift
//  CUDA
//
//  Created by Richard Wei on 11/3/16.
//
//

extension StaticString : Equatable {
    public static func == (lhs: StaticString, rhs: StaticString) -> Bool {
        return lhs.utf8Start == rhs.utf8Start
    }
}

protocol SourceHashable : Equatable, Hashable {
    var source: String { get }
}

extension SourceHashable where Self : RawRepresentable, Self.RawValue == StaticString {
    var source: String {
        return String(bytesNoCopy: UnsafeMutablePointer(mutating: rawValue.utf8Start),
                      length: rawValue.utf8CodeUnitCount,
                      encoding: .utf8,
                      freeWhenDone: false)!
    }
}

/// Kernel source with generic TYPE
///
/// - sum: void sum(T*, SIZE, T*)
/// - asum: void asum(T*, SIZE, T*)
/// - fill: void fill(T*, SIZE)
enum KernelSource: StaticString, SourceHashable {
    case sum = "KN(TYPE *v, SIZE c, TYPE *res){*res=0; for (long i=0; i<c; i++) *res+=v[i];}" // TODO: parallelism
    case asum = "KN(TYPE *v, SIZE c, TYPE *res){*res=0; for (long i=0; i<c; i++) *res+=abs(v[i]);}" // TODO: parallelism
    case fill = "KN(TYPE *v, TYPE x, SIZE c){ID(i); if (i<c) v[i]=x;}"
}

/// Kernel source with generic T and 1-place transformation function FUNC (eg. tan, sin)
///
/// - transform: void transform(T*, SIZE, T*)
enum FunctorialKernelSource: StaticString, SourceHashable {
    case transform = "KN(TYPE *v, SIZE c, TYPE *res){ID(i); if (i<c) res[i]=FUNC(v[i]);}"
}


/// Kernel source with generic TYPE and binary operation OP
///
/// - elementwise: void elementwise(T, T*, T, T*, SIZE, T*)
/// - scalarRight: void scalarRight(T, T*, T, SIZE, T*)
/// - scalarLeft: void scalarRight(T, T, T*, SIZE, T*)
enum BinaryOperationKernelSource: StaticString, SourceHashable {
    case elementwise = "KN(TYPE a, TYPE *x, TYPE b, TYPE *y, SIZE c, TYPE *res) {ID(i); if (i<c) res[i] = OP(a*x[i],b*y[i]);}"
    case scalarRight = "KN(TYPE a, TYPE *x, TYPE rval, SIZE c, TYPE *res) {ID(i); if (i<c) res[i] = OP(a*x[i],rval);}"
    case scalarLeft = "KN(TYPE lval, TYPE a, TYPE *x, SIZE c, TYPE *res) {ID(i); if (i<c) res[i] = OP(lval,a*x[i]);}"
}

extension StaticString : Hashable {
    public var hashValue: Int {
        return utf8Start.hashValue
    }
}
