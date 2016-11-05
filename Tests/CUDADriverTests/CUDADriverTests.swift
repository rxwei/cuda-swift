import XCTest
@testable import CUDADriver
import NVRTC

class CUDADriverTests: XCTestCase {

    override func setUp() {
        Driver.initialize()
    }

    /// Test initialize multiple times
    func testInitialize() {
        Driver.initialize()
        Driver.initialize()
        Driver.initialize()
    }

    func testDeviceCount() {
        XCTAssertGreaterThanOrEqual(Device.count, 1)
    }

    func testDevice() {
        let computability = Device.main!.computeCapability
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

        let program = try Program(source: source, name: "test")
        let ptx = try Compiler.compile(program)
        try Device.main!.withContext { context in
            let module = try Module(ptx: ptx)
            _ = module.function(named: "gIncr")
            _ = module.function(named: "gSum")
            _ = module.function(named: "saxpy")
        }
    }

    static var allTests : [(String, (CUDADriverTests) -> () throws -> Void)] {
        return [
            ("testInitialize", testInitialize),
            ("testDeviceCount", testDeviceCount),
            ("testDevice", testDevice),
            ("testModule", testModule),
        ]
    }
}
