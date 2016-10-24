//import CUDADriver
import Foundation
import CUDARuntime

UnsafeMutableDevicePointer<Float>.self

/*
do {
    try Driver.initialize()

    Device.count

    Device.default.pciBusID

    Device.default.name

    Device.default.properties.maxThreadsDim

    Device.default.properties.totalConstantMemory

    Device.default.computeCapability
    
    let context = try Context(device: Device.default)

    let ptxPath = Bundle.main.path(forResource: "sum", ofType: "ptx")!
    let ptx = try PTX(contentsOfFile: ptxPath)

    let jitModule = try Module(ptx: ptx)

    let kernel = try jitModule.function(named: "sum")

    let arrayOnDevice = try UnsafeMutableDevicePointer<Float>.allocate(capacity: 5)
    try arrayOnDevice.assign(fromHost: [1.0, 2.0, 0.0, 3.0, 0.1])
    
    let result = try UnsafeMutableDevicePointer<Float>.allocate(capacity: 1)

    try arrayOnDevice.withDeviceAddressHandle { arrPtr in
        try result.withDeviceAddressHandle { resPtr in
            try kernel.launch(
                withArguments: [arrPtr, 5, resPtr],
                inGrid: Function.GridSize(x: 1, y: 0, z: 0),
                ofBlocks: Function.BlockSize(x: 1, y: 0, z: 0, sharedMemorySize: 16),
                stream: nil
            )
        }
    }

    try result.deallocate()
    try arrayOnDevice.deallocate()

}

catch {
    print(error)
}
 */
