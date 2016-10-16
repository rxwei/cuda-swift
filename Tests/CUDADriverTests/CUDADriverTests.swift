import XCTest
@testable import CUDADriver

class CUDADriverTests: XCTestCase {

    func testDevice() {
        XCTAssertGreaterThanOrEqual(DeviceManager.devices.count, 1)
        XCTAssertEqual(Device.default.computeCapability, Device.ComputeCapability(major: 3, minor: 0))
    }

    static var allTests : [(String, (CUDADriverTests) -> () throws -> Void)] {
        return [
        ]
    }
}
