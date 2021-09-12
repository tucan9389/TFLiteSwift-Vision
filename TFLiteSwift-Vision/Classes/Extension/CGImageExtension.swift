/*
* Copyright Doyoung Gwak 2020
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

//
//  UIImageExtension.swift
//  PoseEstimation-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/03/21.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import CoreGraphics
import TensorFlowLite

extension CGImage {
    
    public func grayImage(width: Int, height: Int) -> CGImage? {
        let cgImage = self
        guard cgImage.width > 0, cgImage.height > 0 else { return nil }
        
        let size = CGSize(width: width, height: height)

        let bitmapInfo = CGBitmapInfo(
            rawValue: CGImageAlphaInfo.none.rawValue
        )
        
        guard let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: cgImage.bitsPerComponent,
                bytesPerRow: width * 1,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: bitmapInfo.rawValue)
          else {
            return nil
        }
        context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        let outputCGImage = context.makeImage()
        return outputCGImage
    }
    
    /// source: https://github.com/tensorflow/examples/blob/master/lite/examples/digit_classifier/ios/DigitClassifier/TFLiteExtensions.swift
    /// `UIImage` → `CGContext`(resize and make gray data) → `CGImage` → Byte `Array`(and normalization) → `Data`
    public func grayData(
        normalization: TFLiteVisionInterpreter.NormalizationOptions = .none,
        dataType: Tensor.DataType = .float32) -> Data? {
        
        guard let pixelBytes = grayImage(width: width, height: height)?.dataProvider?.data as Data? else { return nil }
        
        let size = CGSize(width: self.width, height: self.height)
        
        switch dataType {
        case .uInt8:
            return Data(copyingBufferOf: pixelBytes.map { UInt8($0) })
        case .float32:
            switch normalization {
            case .none:
                return Data(copyingBufferOf: pixelBytes.map { Float($0) })
            case .scaled(from: let from, to: let to):
                return Data(copyingBufferOf: pixelBytes.map { element -> Float in ((Float(element) * (1.0 / 255.0)) * (to - from)) + from })
            case .meanStd(mean: let mean, std: let std):
                var bytes = pixelBytes.map { Float($0) } // normalization
                for i in 0 ..< Int(size.width * size.height) {
                    bytes[width * height * 0 + i] = (Float32(bytes[i * 1 + 0]) - mean[0]) / std[0] // Gray
                }
                return Data(copyingBufferOf: bytes)
            }
        default:
            fatalError("don't support the type: \(dataType)")
        }
    }
}
