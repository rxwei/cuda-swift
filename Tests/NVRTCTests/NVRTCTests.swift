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

    func testLog() throws {
        let source: String =
            "extern \"C\" __global__ void incr(float *d, size_t ind, float delta) {"
          + "    d[ind] += elta; // undefined variable"
          + "}"
        let program = try Program(source: source, name: "test")
        do {
            let ptx = try Compiler.compile(program)
            XCTAssertEqual(ptx.name, "test")
        }
        catch _ as CompilerError {
            guard let log = program.compilationLog else { XCTFail("Log unavailable"); return }
            XCTAssertTrue(log.contains("error: identifier \"elta\" is undefined\n\nAt end of source: error: expected a \"}\"\n\n2 errors detected in the compilation of"))
        }
    }

    static var allTests : [(String, (NVRTCTests) -> () throws -> Void)] {
        return [
            ("testCompile", testCompile),
            ("testLog", testLog),
        ]
    }
}
