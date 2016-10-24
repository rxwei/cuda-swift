import XCTest
@testable import CUDADriver
@testable import class NVRTC.Compiler

class CUDADriverTests: XCTestCase {

    override func setUp() {
        Driver.initialize()
    }

    func testDeviceCount() {
        XCTAssertGreaterThanOrEqual(Device.count, 1)
    }

    func testDevice() {
        let computability = Device.main.computeCapability
        XCTAssertGreaterThanOrEqual(computability.major, 1)
        XCTAssertGreaterThanOrEqual(computability.minor, 0)
    }

    func testModule() throws {
        let source: String =
            "extern \"C\" __global__ void gIncr(float *d, size_t ind, float delta) {"
          + "    d[ind] += delta;"
          + "}"
          + "extern \"C\" __global__ void gSum(float *d, size_t size, float *total) {"
          + "    total = 0;"
          + "    for (size_t i = 0; i < size; ++i) {"
          + "        *total += d[i];"
          + "    }"
          + "}"
          + "extern \"C\" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n) {"
          + "    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;"
          + "    if (tid < n) out[tid] = a * x[tid] + y[tid];"
          + "}";

        let ptx = try Compiler.compile(source)
        let ctx = Device.main.makeContext()
        let module = try Module(ptx: ptx)
        let function = module.function(named: "gSum")

        let numbers: [Float] = [1, 2, 3, 4, 5]
        var result: Float = 0

        /// TODO: launch kernel

        Context.synchronize()
    }

    static var allTests : [(String, (CUDADriverTests) -> () throws -> Void)] {
        return [
            ("testDeviceCount", testDeviceCount),
            ("testDevice", testDevice),
            ("testModule", testModule),
        ]
    }
}
