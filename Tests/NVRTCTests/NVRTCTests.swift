import XCTest
@testable import NVRTC

class NVRTCTests: XCTestCase {

    func testCompile() {
        let source: String =
            "__global__ void gIncr(float *d, size_t ind, float delta) {"
          + "    d[ind] += delta;"
          + "}"
          + "__global__ void gSum(float *d, size_t size, float *total) {"
          + "    total = 0;"
          + "    for (size_t i = 0; i < size; ++i) {"
          + "        *total += d[i];"
          + "    }"
          + "}"
        /// If we use measure block here, Linux tests will fail under Swift 3.0.1-preview2
        do {
            let ptx = try Compiler.compile(source, named: "sum")
            XCTAssertEqual(ptx.name, "sum")
            /// TODO: Compare PTX
        }
        catch {
            XCTFail(error.localizedDescription)
        }
    }

    static var allTests : [(String, (NVRTCTests) -> () throws -> Void)] {
        return [
            ("testCompile", testCompile),
        ]
    }
}
