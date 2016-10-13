import PackageDescription

let package = Package(
    name: "CUDA",
    dependencies: [
        .Package(url: "https://github.com/rxwei/CCUDA", majorVersion: 1)
    ]
)
