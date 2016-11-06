# cuda-swift

This project provides a native Swift interface to CUDA with the following
modules:

- [x] CUDA Driver API `import CUDADriver`
- [x] CUDA Runtime API `import CUDADriver`
- [x] NVRTC - CUDA Runtime Compiler `import NVRTC`
- [x] cuBLAS - CUDA Basic Linear Algebra Subprograms `import CuBLAS`
- [x] Warp - GPU Acceleration Library ([Thrust](https://github.com/thrust/thrust) counterpart)

Any machine with CUDA 7.0+ and a CUDA-capable GPU is supported. Xcode Playground
is supported as well. Please refer to [Usage](#Usage)
and [Components](#Components).

## Quick look

### Value types

CUDA Driver, Runtime, cuBLAS, and NVRTC (real-time compiler) are wrapped in
native Swift types. Warp provides higher level value types, `DeviceArray` and
`DeviceValue`, with copy-on-write semantics.

```swift
import Warp

/// Initialize two arrays on device
var x: DeviceVector<Float> = [1.0, 2.0, 3.0, 4.0, 5.0]
let y: DeviceVector<Float> = [1.0, 2.0, 3.0, 4.0, 5.0]

/// Scalar multiplication
x *= 2 // x => [2.0, 4.0, 6.0, 8.0, 10.0] on device

/// Addition
x += y // x => [3.0, 6.0, 9.0, 12.0, 15.0] on device

/// Dot product
x • y // => 165.0

/// A chain of operations
x *= x • y // x => [495.0, 990.0, 1485.0, 1980.0, 2475.0] on device

/// Sum
x.reduced() // => 15

/// Absolute sum
x.sumOfAbsoluteValues() // => 15
```

### Real-time compilation

#### Compile source string to PTX
```swift
let source: String =
  + "extern \"C\" __global__ void saxpy(float a, float *x, float *y, float *out, int n) {"
  + "    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;"
  + "    if (tid < n) out[tid] = a * x[tid] + y[tid];"
  + "}";
let ptx = try Compiler.compile(source)
```

#### Load a module from PTX using Driver API within a context
```swift
Device.main.withContext { context in
    let module = try Module(ptx: ptx)
    let function = module.function(named: "saxpy")
    
    var x: DeviceArray<Float> = [1, 2, 3, 4, 5, 6, 7, 8]
    var y: DeviceArray<Float> = [2, 3, 4, 5, 6, 7, 8, 9]
    var result = DeviceArray<Float>(capacity: 8)

    try function<<<(1, 8)>>>[.float(1.0), .array(&x), .array(&y), .array(&result), .int(8)]

    /// result => [3, 5, 7, 9, 11, 13, 15, 17] on device
}
```

## Package Information

Add a dependency:

```swift
.Package(url: "https://github.com/rxwei/cuda-swift", majorVersion: 1)
```

You may use the `Makefile` in this repository for you own project. No extra path
configuration is needed.

Otherwise, specify the path to your CUDA headers and library at `swift build`.

#### macOS
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib
```

#### Linux
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib64
```

## Components

### Core

- [x] CUDADriver - CUDA Driver API
    - [x] `Context`
    - [x] `Device`
    - [x] `Function`
    - [x] `PTX`
    - [x] `Module`
    - [x] `Stream`
    - [x] `UnsafeMutableDevicePointer<T>`
    - [x] `DriverError` (all error codes from CUDA C API)
- [x] CUDARuntime - CUDA Runtime API
    - [x] `UnsafeDevicePointer<T>` and `UnsafeMutableDevicePointer<T>`
    - [x] `Device`
    - [x] `Stream`
    - [x] `RuntimeError` (all error codes from CUDA C API)
- [x] NVRTC - CUDA Runtime Compiler
    - [x] `Compiler`
- [x] CuBLAS - GPU Basic Linear Algebra Subprograms (in-progress)
    - [x] Level 1 BLAS operations
    - [ ] Level 2 BLAS operations
    - [ ] Level 3 BLAS operations
- [x] Warp - GPU Acceleration Library (Thrust counterpart)
    - [x] `DeviceArray<T>` (generic array in device memory)
    - [x] `DeviceValue<T>` (generic value in device memory)
    - [x] Acclerated vector operations ([Thrust](https://github.com/thrust/thrust) counterpart)

### Optional

- [x] Swift Playground
  - CUDADriver works in the playground. But other modules cause the "couldn't lookup
    symbols" problem for which we don't have a solution until Xcode is fixed.
  - To use the playground, open the Xcode workspace file, and add a library for
    every modulemap under `Frameworks`.

### Next steps

TBA

## Dependencies

- [CCUDA (CUDA C System Module)](https://github.com/rxwei/CCUDA)

## License

MIT License

CUDA is a registered trademark of NVIDIA Corporation.
