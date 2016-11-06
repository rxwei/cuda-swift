#!/usr/bin/swift -F. -L. -L/usr/local/cuda/lib -I. -I/usr/local/cuda/include -target x86_64-apple-macosx10.10

import NVRTC
import Warp
import CUDADriver

guard let device = Device.all.first else {
    fatalError("No CUDA device available")
}

/// Data
let n = 1024
var x = DeviceArray<Double>(repeating: 1.0, count: n)
var y = DeviceArray<Double>(fromHost: Array(sequence(first: 0, next: {$0+1}).prefix(n)))
var result = DeviceArray<Double>(capacity: n)

device.withContext { context in
    /// SAXPY: Z = a * X + Y
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
    /// Launch kernel
    try saxpy<<<(n/128, 128)>>>[.int(1024), .double(5.1), .array(&x), .array(&y), .array(&result)]
}

print(result.hostArray)
