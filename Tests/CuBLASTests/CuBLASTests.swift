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
            let result = BLAS.current.sumOfAbsoluteValues(in: xx)
            XCTAssertEqual(result.value, xx.reduce(0, {$0+abs($1)}))
            let xxDouble: DeviceArray<Double> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            let resultDouble = BLAS.current.sumOfAbsoluteValues(in: xxDouble)
            XCTAssertEqual(resultDouble.value, xxDouble.reduce(0, {$0+abs($1)}))
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
            BLAS.current.add(xx, multipliedBy: DeviceValue(0.0002), onto: &yy2)
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
            BLAS.current.add(vectorX, multipliedBy: DeviceValue(1.0), onto: &vectorY)
            print(vectorY.copyToHost())
        }
    }

    func testVectors() {
        var xx: DeviceVector<Float> = [1, 2, 3, 4, 5]
        let yy: DeviceVector<Float> = [1, 2, 3, 4, 5]

        /// Scalar multiplication
        xx *= 2
        XCTAssertEqual(xx.copyToHost(), [2, 4, 6, 8, 10])

        /// Addition
        xx += yy
        XCTAssertEqual(xx.copyToHost(), [3, 6, 9, 12, 15])

        /// Index of max
        XCTAssertEqual(BLAS.current.oneBasedIndexOfMax(in: xx).value, 5)

        /// Dot product
        XCTAssertEqual((xx • yy).value, 165.0)
    }

    static var allTests : [(String, (CuBLASTests) -> () throws -> Void)] {
        return [
            ("testSum", testSum),
            ("testSaxpy", testSum),
            ("testVectors", testVectors),
        ]
    }
}
