//
//  BLASTests.swift
//  CUDA
//
//  Created by Richard Wei on 11/1/16.
//

import XCTest
@testable import Warp

class BLASTests: XCTestCase {

    /*
    func testSum() {
        measure {
            let xx: DeviceArray<Float> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            let result = ∑+(xx)
            XCTAssertEqual(result.value, xx.copyToHost().reduce(0, {$0+abs($1)}))
            let xxDouble: DeviceArray<Double> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            let resultDouble = ∑+(xxDouble)
            XCTAssertEqual(resultDouble.value, xxDouble.copyToHost().reduce(0, {$0+abs($1)}))
        }
    }
 */

    /*
    func testSaxpy() {
        measure {
            let xx: DeviceArray<Float> = [1.2, 3.3, -3, 4.0, 5.6, 7.5, -10, -100.2012432, 20]
            var yy: DeviceArray<Float> = [  1,   2,  3,   4,   5,   6,   7,            8,  9]
            var yy2 = yy
            let yyOrig = yy
            /// Compute addition by BLAS
            BLAS.main.add(xx, onto: &yy)
            BLAS.main.add(xx, multipliedBy: DeviceValue(0.0002), onto: &yy2)
            /// Compute addition using CPU
            let expected = zip(xx, yyOrig).map { x, y in x.value * 1.0 + y.value }
            let expected2 = zip(xx, yyOrig).map { x, y in x.value * 0.0002 + y.value }
            /// Compare
            XCTAssertEqual(yy.copyToHost(), expected)
            XCTAssertEqual(yy2.copyToHost(), expected2)

            /// Another test
            /// We are going to add vector X onto Y using cuBLAS
            let vectorX: DeviceArray<Float> = [1.1, 2.2, 0.3, -4.0]
            var vectorY: DeviceArray<Float> = [0.0, -2.7, 0.009, -0.07]
            BLAS.main.add(vectorX, multipliedBy: DeviceValue(1.0), onto: &vectorY)
            print(vectorY.copyToHost())
        }
    }
    */

    /*
    func testVectors() {
        var xx: DeviceVector<Float> = [1, 2, 3, 4, 5]
        let yy: DeviceVector<Float> = [1, 2, 3, 4, 5]

        /// Scalar multiplication
        xx *= 2
        XCTAssertEqual(xx.copyToHost(), [2, 4, 6, 8, 10])

        /// Addition
        xx += yy
        XCTAssertEqual(xx.copyToHost(), [3, 6, 9, 12, 15])

        /// Index of min/max
        XCTAssertEqual(BLAS.main.oneBasedIndexOfMin(in: xx).value, 1)
        XCTAssertEqual(BLAS.main.oneBasedIndexOfMax(in: xx).value, 5)

        /// Min/max
        XCTAssertEqual(BLAS.main.max(in: xx).value, 15)
        XCTAssertEqual(BLAS.main.min(in: xx).value, 3)

        /// Dot product
        XCTAssertEqual((xx • yy).value, 165.0)

        /// Chain
        xx *= xx • yy
        XCTAssertEqual(xx.copyToHost(), [495.0, 990.0, 1485.0, 1980.0, 2475.0])
    }

    func testMatrix() {
        let matrixElements: [Float] = [
             1, 2, 3, 4, 5,
             5, 4, 3, 2, 1,
             2, 3, 4, 7, 0,
            -1, 2, 3, 0, 2
        ]
        var matrix = DeviceMatrix(rowCount: 4, columnCount: 5, rowMajorElements: matrixElements)
        XCTAssertEqual(matrix[2, 3].value, 7)
        /// Assign vector. Should not copy
        matrix[0] = [3, 4, 5, 6, 7]
        XCTAssertEqual(matrix[0].copyToHost(), [3, 4, 5, 6, 7])

        let vec = matrix[0] // = [3,4,5,6,7]
        matrix[0] = [1, 2, 3, 4, 5]
        XCTAssertEqual(vec.copyToHost(), [3, 4, 5, 6, 7])
        
        var val30 = matrix[3, 0]
        val30.value = 3
        XCTAssertEqual(matrix[3, 0].value, -1)
    }
 */

    static var allTests : [(String, (BLASTests) -> () throws -> Void)] {
        return [
        ]
    }

}
