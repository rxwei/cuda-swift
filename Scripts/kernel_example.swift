#!/usr/bin/swift -F. -L. -L/usr/local/cuda/lib -I. -I/usr/local/cuda/include -target x86_64-apple-macosx10.10

import NVRTC
import CUDARuntime

guard let device = Device.current else {
    fatalError("No CUDA device available")
}

/// This program performs Z = a * X + Y, where a is a scalar and X and Y are vectors.
let saxpySource =
    "extern \"C\" __global__ void saxpy(size_t n, double a, double *x, double *y, double *z) {"
  + "    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;"
  + "    if (tid < n) z[tid] = a * x[tid] + y[tid];"
  + "}"

let module = try Module(source: saxpySource, compileOptions: [
    .computeCapability(device.computeCapability),
    .contractIntoFMAD(false),
    .useFastMath
])

let saxpy = module.function(named: "saxpy")!

/// Data
let n = 1024
var x = DeviceArray<Double>(repeating: 1.0, count: n)
var y = DeviceArray<Double>(fromHost: Array(sequence(first: 0, next: {$0+1}).prefix(n)))
var result = DeviceArray<Double>(capacity: n)

/// Add arguments to a list
var args = ArgumentList()
args.append(Int32(n))    /// count
args.append(Double(5.1)) /// a
args.append(&x)          /// X
args.append(&y)          /// Y
args.append(&result)     /// Z

/// Launch kernel
try saxpy<<<(n/128, 128)>>>(args)

print(result.copyToHost())
