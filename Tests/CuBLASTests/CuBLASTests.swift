import XCTest
@testable import CuBLAS
import CUDARuntime

class CuBLASTests: XCTestCase {

    override func setUp() {
        _ = BLAS.global
    }

    func testSum() {
        measure {
            let xx: DeviceArray<Float> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            let result = BLAS.global.absSum(xx)
            XCTAssertEqual(result, xx.reduce(0, {$0+abs($1)}))
            let xxDouble: DeviceArray<Double> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            let resultDouble = BLAS.global.absSum(xxDouble)
            XCTAssertEqual(resultDouble, xxDouble.reduce(0, {$0+abs($1)}))
        }
    }

    func testSaxpy() {
        measure {
            let xx: DeviceArray<Float> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            var yy: DeviceArray<Float> = [  1,   2,  3,   4,   5,   6,   7,            8,  9]
            var yy2 = yy
            let yyOrig = yy
            /// Compute addition by BLAS
            BLAS.global.add(xx, onto: &yy)
            BLAS.global.add(xx, multipliedBy: 0.0002, onto: &yy2)
            /// Compute addition using CPU
            let expected = zip(xx, yyOrig).map { x, y in x * 1.0 + y }
            let expected2 = zip(xx, yyOrig).map { x, y in x * 0.0002 + y }
            /// Compare
            XCTAssertEqual(yy.makeHostArray(), expected)
            XCTAssertEqual(yy2.makeHostArray(), expected2)
        }
    }

    static var allTests : [(String, (CuBLASTests) -> () throws -> Void)] {
        return [
            ("testSum", testSum),
            ("testSaxpy", testSum),
        ]
    }
}
