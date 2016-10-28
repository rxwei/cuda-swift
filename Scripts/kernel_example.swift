#!/usr/bin/swift -F. -L. -L/usr/local/cuda/lib -I. -I/usr/local/cuda/include -target x86_64-apple-macosx10.10

import CUDARuntime
import NVRTC

let source: String =
    "__global__ void mult(float a, float *x, size_t n) {"
  + "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;"
  + "    if (i < n) x[i] = a * x[i];"
  + "}";

do {

    let ptx = try Compiler.compile(Program(source: source))
    let module = try Module(ptx: ptx)
    let mult = module.function(named: "mult")!
    var n: Int32 = 256
    var x = DeviceArray<Float>(fromHost: sequence(first: Float(0.0), next: {$0+1.0}).prefix(Int(n)).map{$0})
    var a: Float = 5.0

    var args = Function.Arguments()
    args.append(&a)
    args.append(&x)
    args.append(&n)

    print("Before:", x.copyToHost())

    try mult.launch(
        withArguments: &args,
        inGrid: Function.GridSize(x: 8, y: 1, z: 1),
        ofBlocks: Function.BlockSize(x: 32, y: 1, z: 1),
        stream: nil
    )

    print("After:", x.copyToHost())

} catch {
    print(error)
}
