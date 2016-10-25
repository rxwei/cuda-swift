# CUDA for Swift

This project provides a *native* Swift interface to CUDA.

Under active development, not yet ready to use.

## Components

### Core

- [x] NVRTC - CUDA Runtime Compiler
    - [x] `Compiler`
- [x] CUDADriver - CUDA Driver API
    - [x] `Context`
    - [x] `Device`
    - [x] `Function` (kernel)
    - [x] `PTX`    
    - [x] `Module`
    - [x] `Stream`
    - [x] `UnsafeMutableDevicePointer<T>`
    - [x] `DriverError` (all error codes from CUDA C API)
- [x] CUDARuntime - CUDA Runtime API
    - [x] `Device`
    - [x] `UnsafeMutableDevicePointer<T>`
    - [x] `DeviceValue<T>` (generic value in device memory)
    - [x] `DeviceArray<T>` (generic value type array in device memory)
    - [x] `Module`, `Function` bridged from `CUDADriver`
    - [x] `Stream`
    - [x] `RuntimeError` (all error codes from CUDA C API)
- [ ] CuBLAS - GPU Basic Linear Algebra Subprograms (in-progress)
    - [x] Initialization
    - [ ] Level 1 BLAS operations
    - [ ] Level 2 BLAS operations
    - [ ] Level 3 BLAS operations
- [ ] NVBLAS - CPU Basic Linear Algebra Subprograms (We might not need this.)

### Optional

- [x] Swift Playground
  - CUDADriver works in the playground. But other modules cause the "couldn't lookup
    symbols" problem for which we don't have a solution until Xcode is fixed.
  - To use the playground, open the Xcode workspace file, and add a library for
    every modulemap under `Frameworks`.

### Next steps

- [ ] [CuDNN](https://github.com/rxwei/cudnn-swift)

## Quick look

### Value types

CUDA Driver, Runtime, cuBLAS, and NVRTC are wrapped in native Swift types. For
higher level APIs such as runtime and cuBLAS, we wrap the types needed manual
memory allocation into **value types** with copy-on-write semantics. This
enables us to casually perform the following GPU computation without dealing
with manual device memory allocation:

```swift
/// We are going to add vector X onto Y using cuBLAS
let vectorX: DeviceArray<Float> = [1.1, 2.2, 0.3, -4.0]
var vectorY: DeviceArray<Float> = [0.0, -2.7, 0.009, -0.07]
BLAS.current.add(vectorX, multipliedBy: 1.0, onto: &vectorY)
print(vectorY.copyToHost())
```

With value types, we can worry less about reference to device memory. The following
example demonstrates copy-on-write.
```swift
let vectorX: DeviceArray<Float> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
var vectorY: DeviceArray<Float> = [  1,   2,  3,   4,   5,   6,   7,            8,  9]
/// Make a value reference to vector Y. This does not copy anything!
let originalVectorY = vectorY
/// Mutate Y by adding  onto vectorY 
BLAS.current.add(vectorX, onto: &vectorY)
/// After vectorY's mutation, originalVectorY still has the original value we assigned 
/// in line 2, and vectorY is no longer equal to originalVectorY. Because DeviceArray
/// made a copy of vectorY's orignal buffer when mutated.
```

### Real-time compilation

In C/C++, we can write all the logic in a *.cu file. How do we do that in Swift? 

Swift Package Manager does not support invoking external build systems, but this is
not a limitation at all! Real-time compilation is convenient and should not be a
performance bottleneck for GPU-heavy applications. We can simply compile *.cu files 
and *.ptx files at runtime using NVRTC. The following code compiles the CUDA kernel
into PTX, and loads it to device memory.

#### Compile source string to PTX
```swift
let source: String =
  + "extern \"C\" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n) {"
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
    function.launch(withArguments: ..., inGrid: ..., ofBlocks: ..., stream: ...) 
}
```
#### Alternatively, using Runtime API without dealing with context
```swift
let module = try Module(ptx: ptx)
let function = module.function(named: "saxpy")
function.launch(withArguments: ..., inGrid: ..., ofBlocks: ..., stream: ...) 
```

## Dependencies

- [CCUDA](https://github.com/rxwei/CCUDA)

## Usage

After its initial release, you'll be able to use cuda-swift by adding the
following dependency: 

```swift
.Package(url: "https://github.com/rxwei/cuda-swift", majorVersion: 1)
```

You'll need to specify the path to your CUDA headers and library at `swift build`.

#### macOS
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib
```

#### Linux
```
swift build -Xcc -I/usr/lib/cuda/include -Xlinker -L/usr/lib/cuda/lib
```

You may use the Makefile in this project.
