import XCTest
@testable import CUDARuntime
import NVRTC
import class CUDADriver.Stream

class CUDARuntimeTests: XCTestCase {

    func testDevice() {
        let computability = CUDARuntime.Device.current.computeCapability
        XCTAssertGreaterThanOrEqual(computability.major, 1)
        XCTAssertGreaterThanOrEqual(computability.minor, 0)
    }

    func testPointer() {
        measure {
            let localArray: ContiguousArray = [1, 2, 3, 4, 5, 6, 7, 8]
            let pointer = UnsafeMutableDevicePointer<Int>.allocate(capacity: 8)
            pointer.assign(fromHost: localArray)
            XCTAssertEqual(pointer.load(), 1)
            for i in localArray.indices {
                XCTAssertEqual(localArray[i], pointer[i])
            }
            /// Add one to each device element
            for i in localArray.indices {
                pointer[i] += 1
                XCTAssertEqual(localArray[i] + 1, pointer[i])
            }
            pointer.deallocate()
        }
    }

    func testArray() {
        let hostArray: [Int] = [1, 2, 3, 4, 5]
        /// Array literal initialization!
        let devArray: DeviceArray<Int> = [1, 2, 3, 4, 5]
        XCTAssertEqual(hostArray, Array(devArray))
        let hostArrayFromDev: [Int] = devArray.copyToHost()
        XCTAssertEqual(hostArray, hostArrayFromDev)

        /// Test copy-on-write
        var devArray2 = devArray
        var devArray3 = devArray
        let devArray4 = devArray3
        devArray2[0].value = 3
        XCTAssertNotEqual(Array(devArray), Array(devArray2))
        devArray2[0] = DeviceValue(1)
        XCTAssertEqual(Array(devArray), Array(devArray2))
        devArray3[0].value = 4
        var val3_0 = devArray3[0]
        var origVal3_0 = val3_0
        XCTAssertEqual(val3_0.value, 4)
        val3_0.value = 10
        XCTAssertEqual(val3_0.value, 10)
        XCTAssertEqual(origVal3_0.value, 4)
        var devArray5 = devArray
        let val5_0 = devArray5[0]
        devArray5[0].value = 100
        XCTAssertEqual(val5_0.value, 1)
        devArray5[0] = DeviceValue(100)
        XCTAssertEqual(val5_0.value, 1)
        XCTAssertEqual(devArray5[0].value, 100)
        XCTAssertNotEqual(Array(devArray2), Array(devArray3))
        XCTAssertEqual(devArray.copyToHost(), Array(devArray))
        XCTAssertEqual(devArray.copyToHost(), [1, 2, 3, 4, 5])
        XCTAssertEqual(devArray2.copyToHost(), [1, 2, 3, 4, 5])
        XCTAssertEqual(devArray3.copyToHost(), [4, 2, 3, 4, 5])
        XCTAssertEqual(devArray4.copyToHost(), [1, 2, 3, 4, 5])

        /// Array slices
        var devArray6 = devArray // 1...5
        let devArray6_13 = devArray6[1...3]
        XCTAssertEqual(devArray6_13.copyToHost(), [2, 3, 4])
        devArray6[1].value = 20
        XCTAssertEqual(devArray6_13.copyToHost(), [2, 3, 4])
        XCTAssertEqual(devArray6.copyToHost(), [1, 20, 3, 4, 5])

        /// Array value reference
        var V: DeviceArray<Float> = [1, 2, 3]
        let x = V[2]
        XCTAssertEqual(x.value, 3)
        V[2] = DeviceValue(0)
        XCTAssertEqual(x.value, 3)
        V[2].value = 100
        XCTAssertEqual(x.value, 3)

        /// Nested device array literal
        var VV: DeviceArray<DeviceArray<DeviceArray<Float>>> = [
            [[1, 0], [1, 2], [1, 3], [1, 4], [1, 5]],
            [[1, 2], [1, 2], [1, 3], [1, 4], [1, 5]],
        ]
        XCTAssertEqual(VV[0][1].copyToHost(), [1, 2])
        XCTAssertEqual(VV[1][4].copyToHost(), [1, 5])
        let row1: [[Float]] = VV[1].copyToHost()
        let row1ShouldBe: [[Float]] = [[1, 2], [1, 2], [1, 3], [1, 4], [1, 5]]
        XCTAssertTrue(row1.elementsEqual(row1ShouldBe, by: { (xx, yy) in
            xx.elementsEqual(yy)
        }))

        /// Nested array reference literal
        /// Currently mutation FAILS
        var VV1: DeviceArray<DeviceArray<Float>> = {
            let vv1_0: DeviceArray<Float> = [1, 2, 3]
            let vv1_1: DeviceArray<Float> = [4, 5, 6]
            return [ vv1_0, vv1_1 ]
        }()
        XCTAssertEqual(VV1[0].copyToHost(), [1, 2, 3])
        XCTAssertEqual(VV1[1].copyToHost(), [4, 5, 6])
        XCTAssertEqual(VV1[0][0].value, 1)
        XCTAssertEqual(VV1[0][1].value, 2)
        XCTAssertEqual(VV1[0][2].value, 3)
        XCTAssertEqual(VV1[1][0].value, 4)
        XCTAssertEqual(VV1[1][1].value, 5)
        XCTAssertEqual(VV1[1][2].value, 6)
    }

    func testValue() {
        var val = DeviceValue<Int>(1)
        XCTAssertEqual(val.value, 1)
        var val2 = val
        val2.value = 10
        XCTAssertEqual(val.value, 1)
        XCTAssertEqual(val2.value, 10)

        /// Test memory mutation
        val.withUnsafeMutableDevicePointer { ptr in
            ptr.assign(100)
        }
        XCTAssertEqual(val.value, 100)
        XCTAssertNotEqual(val2.value, val.value)

        /// Test CoW memory mutation
        var val3 = val
        val3.withUnsafeMutableDevicePointer { ptr in
            ptr.assign(1000)
        }
        XCTAssertEqual(val3.value, 1000)
        XCTAssertNotEqual(val3.value, val.value)
    }
    
    func testModule() throws {
        let source: String =
            "__global__ void mult(float a, float *x, size_t n) {"
          + "    size_t i = blockIdx.x * blockDim.x + threadIdx.x;"
          + "    if (i < n) x[i] = a * x[i];"
          + "}";
        let ptx = try Compiler.compile(
            Program(source: source),
            options: [
                .gpu(withComputeCapability: Device.current.computeCapability),
                .contractIntoFMAD(true),
            ]
        )
        let module = try Module(ptx: ptx)
        let mult = module.function(named: "mult")!
        var n: Int32 = 256
        var x = DeviceArray<Float>(fromHost: Array(sequence(first: 0.0, next: {$0+1}).prefix(256)))
        var a: Float = 5.0
        
        var args = Function.ArgumentList()
        args.append(&a)
        args.append(&x)
        args.append(&n)
        
        try mult.launch(
            withArguments: &args,
            inGrid: Function.GridSize(x: 8, y: 1, z: 1),
            ofBlocks: Function.BlockSize(x: 32, y: 1, z: 1),
            stream: nil
        )
        
        XCTAssertEqual(x.copyToHost(), [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 230.0, 235.0, 240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 275.0, 280.0, 285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0, 340.0, 345.0, 350.0, 355.0, 360.0, 365.0, 370.0, 375.0, 380.0, 385.0, 390.0, 395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0, 445.0, 450.0, 455.0, 460.0, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0, 550.0, 555.0, 560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0, 600.0, 605.0, 610.0, 615.0, 620.0, 625.0, 630.0, 635.0, 640.0, 645.0, 650.0, 655.0, 660.0, 665.0, 670.0, 675.0, 680.0, 685.0, 690.0, 695.0, 700.0, 705.0, 710.0, 715.0, 720.0, 725.0, 730.0, 735.0, 740.0, 745.0, 750.0, 755.0, 760.0, 765.0, 770.0, 775.0, 780.0, 785.0, 790.0, 795.0, 800.0, 805.0, 810.0, 815.0, 820.0, 825.0, 830.0, 835.0, 840.0, 845.0, 850.0, 855.0, 860.0, 865.0, 870.0, 875.0, 880.0, 885.0, 890.0, 895.0, 900.0, 905.0, 910.0, 915.0, 920.0, 925.0, 930.0, 935.0, 940.0, 945.0, 950.0, 955.0, 960.0, 965.0, 970.0, 975.0, 980.0, 985.0, 990.0, 995.0, 1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0, 1030.0, 1035.0, 1040.0, 1045.0, 1050.0, 1055.0, 1060.0, 1065.0, 1070.0, 1075.0, 1080.0, 1085.0, 1090.0, 1095.0, 1100.0, 1105.0, 1110.0, 1115.0, 1120.0, 1125.0, 1130.0, 1135.0, 1140.0, 1145.0, 1150.0, 1155.0, 1160.0, 1165.0, 1170.0, 1175.0, 1180.0, 1185.0, 1190.0, 1195.0, 1200.0, 1205.0, 1210.0, 1215.0, 1220.0, 1225.0, 1230.0, 1235.0, 1240.0, 1245.0, 1250.0, 1255.0, 1260.0, 1265.0, 1270.0, 1275.0])
    }

    static var allTests : [(String, (CUDARuntimeTests) -> () throws -> Void)] {
        return [
            ("testDevice", testDevice),
            ("testPointer", testPointer),
            ("testArray", testArray),
            ("testValue", testValue),
            ("testModule", testModule)
        ]
    }
}
