import PackageDescription

let package = Package(
    name: "CUDA",
    targets: [
        Target(name: "CUDADriver"),
        Target(name: "CUDARuntime", dependencies: [ "CUDADriver" ]),
        Target(name: "NVRTC", dependencies: [ "CUDADriver" ]),
        Target(name: "CuBLAS", dependencies: [ "CUDADriver", "CUDARuntime" ]),
        Target(name: "Warp", dependencies: [ "CUDADriver", "CUDARuntime", "CuBLAS" ])
    ],
    dependencies: [
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1, minor: 4)
    ]
)

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
