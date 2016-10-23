import XCTest
@testable import CUDARuntime

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
        let hostArrayFromDev: [Int] = devArray.makeHostArray()
        XCTAssertEqual(hostArray, hostArrayFromDev)

        /// Test copy-on-write
        var devArray2 = devArray
        var devArray3 = devArray
        devArray2[0] = 3
        XCTAssertNotEqual(Array(devArray), Array(devArray2))
        devArray3[0] = 4
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
    }

    static var allTests : [(String, (CUDARuntimeTests) -> () throws -> Void)] {
        return [
            ("testDevice", testDevice),
            ("testPointer", testPointer),
            ("testArray", testArray)
        ]
    }
}
