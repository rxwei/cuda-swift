//
//  Memory.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public enum SharedMemoryBankSize: UInt32 {
    case `default` = 0x00
    case fourBytes = 0x01
    case eightBytes = 0x02
}

/// TODO: DevicePointer
