import XCTest
@testable import CUDARuntime
import NVRTC

class CUDARuntimeTests: XCTestCase {

    func testDevice() {
        let computability = CUDARuntime.Device.current.computeCapability
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


    static var allTests : [(String, (CUDARuntimeTests) -> () throws -> Void)] {
        return [
            ("testDevice", testDevice),
            ("testPointer", testPointer),
        ]
    }
}
