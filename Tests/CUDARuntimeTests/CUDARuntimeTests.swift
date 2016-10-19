import XCTest
@testable import CUDARuntime

class CUDARuntimeTests: XCTestCase {

    func testPointer() {
        let pointer = UnsafeMutableDevicePointer<Int>.allocate(capacity: 8)
        pointer.initialize(from: [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(pointer.pointee, 1)
        pointer.deallocate()
    }

    static var allTests : [(String, (CUDARuntimeTests) -> () throws -> Void)] {
        return [
        ]
    }
}
