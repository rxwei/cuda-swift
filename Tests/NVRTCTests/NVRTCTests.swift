import XCTest
@testable import NVRTC

class NVRTCTests: XCTestCase {

    func testCompile() {
        let source: String =
            "extern \"C\" __global__ void incr(float *d, size_t ind, float delta) {"
          + "    d[ind] += delta;"
          + "}"
        do {
            let program = try Program(source: source, name: "test")
            let ptx = try Compiler.compile(program)
            XCTAssertEqual(ptx.name, "test")
            /// TODO: Compare PTX
        }
        catch {
            XCTFail(String(describing: error))
        }
    }

    static var allTests : [(String, (NVRTCTests) -> () throws -> Void)] {
        return [
            ("testCompile", testCompile),
        ]
    }
}
