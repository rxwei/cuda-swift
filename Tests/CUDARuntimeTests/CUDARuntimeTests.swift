import XCTest
@testable import CUDARuntime
import NVRTC

class CUDARuntimeTests: XCTestCase {

    func testDevice() {
        let computability = Device.current.computeCapability
        XCTAssertGreaterThanOrEqual(computability.major, 1)
        XCTAssertGreaterThanOrEqual(computability.minor, 0)
    }

    func testPointer() {
        measure {
            let localArray: ContiguousArray = [1, 2, 3, 4, 5, 6, 7, 8]
            let pointer = UnsafeMutableDevicePointer<Int>.allocate(capacity: 8)
            pointer.assign(fromHost: localArray)
            XCTAssertEqual(pointer.load(), 1)
            for i in localArray.indices {
                XCTAssertEqual(localArray[i], pointer[i])
            }
            /// Add one to each device element
            for i in localArray.indices {
                pointer[i] += 1
                XCTAssertEqual(localArray[i] + 1, pointer[i])
            }
            pointer.deallocate()
        }
    }

    func testArray() {
        let hostArray: [Int] = [1, 2, 3, 4, 5]
        /// Array literal initialization!
        let devArray: DeviceArray<Int> = [1, 2, 3, 4, 5]
        XCTAssertEqual(hostArray, Array(devArray))
        let hostArrayFromDev: [Int] = devArray.copyToHost()
        XCTAssertEqual(hostArray, hostArrayFromDev)

        /// Test copy-on-write
        var devArray2 = devArray
        var devArray3 = devArray
        devArray2[0].value = 3
        XCTAssertNotEqual(Array(devArray), Array(devArray2))
        devArray3[0].value = 4
        XCTAssertNotEqual(Array(devArray2), Array(devArray3))
        XCTAssertEqual(Array(devArray), Array(devArray))
        XCTAssertEqual(Array(devArray), [1, 2, 3, 4, 5])
        XCTAssertEqual(Array(devArray2), [3, 2, 3, 4, 5])
        XCTAssertEqual(Array(devArray3), [4, 2, 3, 4, 5])
    }

    func testValue() {
        var val = DeviceValue<Int>(1)
        XCTAssertEqual(val.value, 1)
        var val2 = val
        val2.value = 10
        XCTAssertEqual(val.value, 1)
        XCTAssertEqual(val2.value, 10)

        /// Test memory mutation
        val.withUnsafeMutableDevicePointer { ptr in
            ptr.assign(100)
        }
        XCTAssertEqual(val.value, 100)
        XCTAssertNotEqual(val2.value, val.value)

        /// Test CoW memory mutation
        var val3 = val
        val3.withUnsafeMutableDevicePointer { ptr in
            ptr.assign(1000)
        }
        XCTAssertEqual(val3.value, 1000)
        XCTAssertNotEqual(val3.value, val.value)
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
        let module = try Module(ptx: ptx)
        _ = module.function(named: "gIncr")
        _ = module.function(named: "gSum")
        _ = module.function(named: "saxpy")
    }

    static var allTests : [(String, (CUDARuntimeTests) -> () throws -> Void)] {
        return [
            ("testDevice", testDevice),
            ("testPointer", testPointer),
            ("testArray", testArray),
            ("testValue", testValue),
            ("testModule", testModule)
        ]
    }
}
