import XCTest
@testable import CUDADriverTests
@testable import CUDARuntimeTests
@testable import CuBLASTests
@testable import NVRTCTests
@testable import CUDATests
@testable import WarpTests

XCTMain([
     testCase(CUDADriverTests.allTests),
     testCase(CUDARuntimeTests.allTests),
     testCase(CuBLASTests.allTests),
     testCase(NVRTCTests.allTests),
     testCase(CUDATests.allTests),
     testCase(WarpTests.allTests),
])
