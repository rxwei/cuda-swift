import XCTest
@testable import CuBLAS
import CUDARuntime

class CuBLASTests: XCTestCase {

    override func setUp() {
        _ = BLAS.current
    }

    func testSum() {
        measure {
            let xx: DeviceArray<Float> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            let result = BLAS.current.absSum(xx)
            XCTAssertEqual(result, xx.reduce(0, {$0+abs($1)}))
            let xxDouble: DeviceArray<Double> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            let resultDouble = BLAS.current.absSum(xxDouble)
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
            BLAS.current.add(xx, onto: &yy)
            BLAS.current.add(xx, multipliedBy: 0.0002, onto: &yy2)
            /// Compute addition using CPU
            let expected = zip(xx, yyOrig).map { x, y in x * 1.0 + y }
            let expected2 = zip(xx, yyOrig).map { x, y in x * 0.0002 + y }
            /// Compare
            XCTAssertEqual(yy.copyToHost(), expected)
            XCTAssertEqual(yy2.copyToHost(), expected2)

            /// Another test
            /// We are going to add vector X onto Y using cuBLAS
            let vectorX: DeviceArray<Float> = [1.1, 2.2, 0.3, -4.0]
            var vectorY: DeviceArray<Float> = [0.0, -2.7, 0.009, -0.07]
            BLAS.current.add(vectorX, multipliedBy: 1.0, onto: &vectorY)
            print(vectorY.copyToHost())
        }
    }

    static var allTests : [(String, (CuBLASTests) -> () throws -> Void)] {
        return [
            ("testSum", testSum),
            ("testSaxpy", testSum),
        ]
    }
}
