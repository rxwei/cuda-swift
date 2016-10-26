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
        let devArray4 = devArray3
        devArray2[0].value = 3
        XCTAssertNotEqual(Array(devArray), Array(devArray2))
        devArray2[0] = DeviceValue(1)
        XCTAssertEqual(Array(devArray), Array(devArray2))
        devArray3[0].value = 4
        var val3_0 = devArray3[0]
        var origVal3_0 = val3_0
        XCTAssertEqual(val3_0.value, 4)
        val3_0.value = 10
        XCTAssertEqual(val3_0.value, 10)
        XCTAssertEqual(origVal3_0.value, 4)
        var devArray5 = devArray
        let val5_0 = devArray5[0]
        devArray5[0].value = 100
        XCTAssertEqual(val5_0.value, 1)
        devArray5[0] = DeviceValue(100)
        XCTAssertEqual(val5_0.value, 1)
        XCTAssertEqual(devArray5[0].value, 100)
        XCTAssertNotEqual(Array(devArray2), Array(devArray3))
        XCTAssertEqual(devArray.copyToHost(), Array(devArray))
        XCTAssertEqual(devArray.copyToHost(), [1, 2, 3, 4, 5])
        XCTAssertEqual(devArray2.copyToHost(), [1, 2, 3, 4, 5])
        XCTAssertEqual(devArray3.copyToHost(), [4, 2, 3, 4, 5])
        XCTAssertEqual(devArray4.copyToHost(), [1, 2, 3, 4, 5])

        /// Test array slices
        var devArray6 = devArray // 1...5
        let devArray6_13 = devArray6[1...3]
        XCTAssertEqual(devArray6_13.copyToHost(), [2, 3, 4])
        devArray6[1].value = 20
        XCTAssertEqual(devArray6_13.copyToHost(), [2, 3, 4])
        XCTAssertEqual(devArray6.copyToHost(), [1, 20, 3, 4, 5])

        /// Test array value reference
        var V: DeviceArray<Float> = [1, 2, 3]
        let x = V[2]
        XCTAssertEqual(x.value, 3)
        V[2] = DeviceValue(0)
        XCTAssertEqual(x.value, 3)
        V[2].value = 100
        XCTAssertEqual(x.value, 3)
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
            "__global__ void gIncr(float *d, size_t ind, float delta) {"
          + "    d[ind] += delta;"
          + "}"
          + "__global__ void gSum(float *d, size_t size, float *total) {"
          + "    total = 0;"
          + "    for (size_t i = 0; i < size; ++i)"
          + "        *total += d[i];"
          + "}"
          + "__global__ void saxpy(float a, float *x, float *y, float *out, size_t n) {"
          + "    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;"
          + "    if (tid < n) out[tid] = a * x[tid] + y[tid];"
          + "}";

        let ptx = try Compiler.compile(Program(source: source))
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
