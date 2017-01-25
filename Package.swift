import PackageDescription

let package = Package(
    name: "CUDA",
    targets: [
        Target(name: "CUDADriver"),
        Target(name: "CUDARuntime", dependencies: [ "CUDADriver" ]),
        Target(name: "NVRTC", dependencies: [ "CUDADriver" ]),
        Target(name: "CuBLAS", dependencies: [ "CUDADriver", "CUDARuntime" ]),
        Target(name: "Warp", dependencies: [ "CUDADriver", "CUDARuntime", "CuBLAS", "NVRTC" ])
    ],
    dependencies: [
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1, minor: 4)
    ]
)

/// Product specification works differently (more formally) in Swift 3.1+
#if (os(Linux) || os(macOS)) && !swift(>=3.1)

let dylib = Product(
    name: "CUDA",
    type: .Library(.Dynamic),
    modules: [ "CUDADriver", "CUDARuntime", "NVRTC", "CuBLAS", ]
)
products.append(dylib)
let a = Product(
    name: "CUDA",
    type: .Library(.Static),
    modules: [ "CUDADriver", "CUDARuntime", "NVRTC", "CuBLAS", ]
)
products.append(a)


let warpA = Product(
    name: "Warp",
    type: .Library(.Static),
    modules: [ "Warp" ]
)
products.append(warpA)

#endif
