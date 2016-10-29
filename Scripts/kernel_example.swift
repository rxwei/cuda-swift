#!/usr/bin/swift -F. -L. -L/usr/local/cuda/lib -I. -I/usr/local/cuda/include -target x86_64-apple-macosx10.10

import NVRTC
import CUDARuntime

let source: String =
    "extern \"C\" __global__ void mult(double *x, double a, size_t n) {"
  + "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;"
  + "    if (i < n) x[i] = a * x[i];"
  + "}";
let ptx = try Compiler.compile(
    Program(source: source),
    options: [
        .computeCapability(Device.current.computeCapability),
        .cpp11,
        .lineInfo,
        .contractIntoFMAD(true),
    ]
)
let module = try Module(ptx: ptx)
let mult = module.function(named: "mult")!
var x = DeviceArray<Double>(fromHost: Array(sequence(first: 0.0, next: {$0+1}).prefix(256)))
var a: Double = 5.0
var n: Int32 = 256

var args = ArgumentList()
args.append(&x)
args.append(&a)
args.append(&n)

print(x.copyToHost())

try mult<<<(8, 32)>>>(args)

print(x.copyToHost())


