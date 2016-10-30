import PackageDescription

let package = Package(
    name: "CUDA",
    targets: [
        Target(name: "CUDADriver"),
        Target(name: "CUDARuntime", dependencies: [ "CUDADriver" ]),
        Target(name: "NVRTC", dependencies: [ "CUDADriver" ]),
        Target(name: "CuBLAS", dependencies: [ "CUDARuntime" ])
    ],
    dependencies: [
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1)
    ]
)

//#if os(macOS)
let dylib = Product(
    name: "CUDA",
    type: .Library(.Dynamic),
    modules: [ "CUDADriver", "CUDARuntime", "NVRTC", "CuBLAS" ]
)
products.append(dylib)
let staticLib = Product(
    name: "CUDA",
    type: .Library(.Static),
    modules: [ "CUDADriver", "CUDARuntime", "NVRTC", "CuBLAS" ]
)
products.append(staticLib)
//#endif
