import PackageDescription

let package = Package(
    name: "CUDA",
    targets: [
        Target(name: "CUDADriver"),
        Target(name: "CUDARuntime"),
        Target(name: "NVRTC", dependencies: [ "CUDADriver" ]),
        Target(name: "CuBLAS", dependencies: [])
    ],
    dependencies: [
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1)
    ]
)
