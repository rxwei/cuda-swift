//
//  Kernels.swift
//  CUDA
//
//  Created by Richard Wei on 11/3/16.
//
//

let builtinKernelSource =
    "extern \"C\" __global__ void Int_sum(long long vector[], size_t count, long long *result) {" +
    "   *result = 0;" +
    "   for (size_t i = 0; i < count; i++)" +
    "       *result += vector[i];" +
    "}" +
    "extern \"C\" __global__ void Float_sum(float vector[], size_t count, float *result) {" +
    "   *result = 0;" +
    "   for (size_t i = 0; i < count; i++)" +
    "       *result += vector[i];" +
    "}" +
    "extern \"C\" __global__ void Double_sum(double vector[], size_t count, double *result) {" +
    "   *result = 0;" +
    "   for (size_t i = 0; i < count; i++)" +
    "       *result += vector[i];" +
    "}"
