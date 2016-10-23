import XCTest
@testable import CUDADriver
@testable import class NVRTC.Compiler

class CUDADriverTests: XCTestCase {

    override func setUp() {
        try! Driver.initialize()
    }

    func testDeviceCount() {
        XCTAssertGreaterThanOrEqual(Device.count, 1)
    }

    func testDevice() {
        let computability = Device.default.computeCapability
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

        let ptx = try PTX(compilingSource: source, named: "test")
        let ctx = try Context(device: Device.default)
        let module = try Module(ptx: ptx)
        let function = try module.function(named: "gSum")

        let numbers: [Float] = [1, 2, 3, 4, 5]
        var result: Float = 0

        Context.synchronize()
    }

    static var allTests : [(String, (CUDADriverTests) -> () throws -> Void)] {
        return [
            ("testDevice", testDevice),
            ("testModule", testModule),
        ]
    }
}
