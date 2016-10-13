import XCTest
@testable import CUDA

class CUDATests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        XCTAssertEqual(CUDA().text, "Hello, World!")
    }


    static var allTests : [(String, (CUDATests) -> () throws -> Void)] {
        return [
            ("testExample", testExample),
        ]
    }
}
