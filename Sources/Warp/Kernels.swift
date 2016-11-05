//
//  Kernels.swift
//  CUDA
//
//  Created by Richard Wei on 11/3/16.
//
//

let kernelSources = [
    "extern \"C\" __global__ void sum(TYPE vector[], size_t count, TYPE *result) {" +
    "   *result = 0;" +
    "   for (size_t i = 0; i < count; i++)" +
    "       *result += vector[i];" +
    "}"
]
