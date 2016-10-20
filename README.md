# CUDA for Swift

This project provides a *native* Swift interface to CUDA.

Under active development, not yet ready to use.

## Components

- [x] CUDADriver - CUDA Driver
- [ ] CUDARuntime - CUDA Runtime (in-progress)
- [ ] CuBLAS - GPU Basic Linear Algebra Subprograms (in-progress)
- [ ] NVBLAS - CPU Basic Linear Algebra Subprograms
- [x] NVRTC - CUDA Runtime Compiler

## Dependencies

- [CCUDA](https://github.com/rxwei/CCUDA)

## Usage

Add a dependency:
```swift
.Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1)
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
