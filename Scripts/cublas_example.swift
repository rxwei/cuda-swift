#!/usr/bin/swift -F. -L. -L/usr/local/cuda/lib -I. -I/usr/local/cuda/include -target x86_64-apple-macosx10.10

import CuBLAS

var vX: DeviceVector<Double> = [1.0, 2.0, 3.0, 4.0, 5.0]
let vY: DeviceVector<Double> = [5.0, 4.0, 3.0, 2.0, 1.0]

vX *= 0.1
print(vX.copyToHost())

vX += vY
print(vX.copyToHost())

let dotProduct = vX • vY
print(dotProduct.value)

