//
//  WarpTests.swift
//  Warp
//
//  Created by Richard Wei on 11/1/16.
//
//

import XCTest
@testable import Warp

class WarpTests: XCTestCase {

    func testLoad() {
        _ = Device.all.map{KernelManager.main(on: $0)}
    }

    static var allTests : [(String, (WarpTests) -> () throws -> Void)] {
        return [
        ]
    }
    
}
