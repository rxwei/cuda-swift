import XCTest
@testable import CUDARuntime

class CUDARuntimeTests: XCTestCase {

    func testPointer() {
        measure {
            let localArray = [1, 2, 3, 4, 5, 6, 7, 8]
            let pointer = UnsafeMutableDevicePointer<Int>.allocate(capacity: 8)
            pointer.initialize(from: localArray)
            XCTAssertEqual(pointer.pointee, 1)
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
        ]
    }
}
