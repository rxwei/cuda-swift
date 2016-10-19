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
        measure {
            do {
                let ptx = try Compiler.compileSource(source, named: "sum")
                XCTAssertEqual(ptx.name, "sum")
                /// TODO: Compare PTX
            }
            catch {
                XCTFail(error.localizedDescription)
            }
        }
    }

    static var allTests : [(String, (NVRTCTests) -> () throws -> Void)] {
        return [
        ]
    }
}
