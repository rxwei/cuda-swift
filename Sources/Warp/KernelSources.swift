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
/// - sum: void sum(T*, long long, T*)
/// - asum: void asum(T*, long long, T*)
/// - fill: void fill(T*, long long)
enum KernelSource: StaticString, SourceHashable {
    case sum = "KERNEL(TYPE *v,long long c,TYPE *res){*res=0; for (long i=0; i<c; i++) *res+=v[i];}"
    case asum = "KERNEL(TYPE *v,long long c,TYPE *res){*res=0; for (long i=0; i<c; i++) *res+=abs(v[i]);}"
    case fill = "KERNEL(TYPE *v,TYPE x,long long c){IDX(i); if (i<c) v[i]=x;}"
}

/// Kernel source with generic TYPE and 1-place transformation function FUNC (eg. tan, sin)
///
/// - transform: void transform(T*, long long, T*)
enum FunctorialKernelSource: StaticString, SourceHashable {
    case transform = "KERNEL(TYPE *v,long long c,TYPE *res){IDX(i); if (i<c) res[i]=FUNC(v[i]);}"
}


/// Kernel source with generic TYPE and binary operation OP
///
/// - elementwise: void elementwise(T, T*, T, T*, long long, T*)
/// - scalarRight: void scalarRight(T, T*, T, long long, T*)
/// - scalarLeft: void scalarRight(T, T, T*, long long, T*)
enum BinaryOperationKernelSource: StaticString, SourceHashable {
    case elementwise = "KERNEL(TYPE a,TYPE *x,TYPE b,TYPE *y,long long c,TYPE *res){IDX(i); if (i<c) res[i]=OP(a*x[i],b*y[i]);}"
    case scalarRight = "KERNEL(TYPE a,TYPE *x,TYPE rval,long long c,TYPE *res){IDX(i); if (i<c) res[i]=OP(a*x[i],rval);}"
    case scalarLeft = "KERNEL(TYPE lval,TYPE a,TYPE *x,long long c,TYPE *res){IDX(i); if (i<c) res[i]=OP(lval,a*x[i]);}"
}

extension StaticString : Hashable {
    public var hashValue: Int {
        return utf8Start.hashValue
    }
}
