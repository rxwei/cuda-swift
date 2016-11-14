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

enum KernelSource: StaticString {
    case sum = "extern \"C\" __global__ void sum(const TYPE *vector, long long count, TYPE *result) { *result = 0; for (long i = 0; i < count; i++) *result += vector[i]; }"
    case asum = "extern \"C\" __global__ void asum(const TYPE *vector, long long count, TYPE *result) { *result = 0; for (long i = 0; i < count; i++) *result += abs(vector[i]); }"
    case axpy = "extern \"C\" __global__ void axpy(TYPE a, const TYPE *x, TYPE *y, long long count) { size_t tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid < count) y[tid] += a * x[tid]; }"
    case scale = "extern \"C\" __global__ void scale(TYPE *vector, TYPE alpha, long long count) { size_t tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid < count) vector[tid] *= alpha; }"
}

extension KernelSource : Hashable {
    var hashValue: Int {
        return rawValue.utf8Start.hashValue
    }
}
