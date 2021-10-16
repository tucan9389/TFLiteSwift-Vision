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

import UIKit
import TensorFlowLite

extension UIImage {
    func pixelBufferFromImage() -> CVPixelBuffer {
        let ciimage = CIImage(image: self)
        //let cgimage = convertCIImageToCGImage(inputImage: ciimage!)
        let tmpcontext = CIContext(options: nil)
        let cgimage =  tmpcontext.createCGImage(ciimage!, from: ciimage!.extent)
        
        let cfnumPointer = UnsafeMutablePointer<UnsafeRawPointer>.allocate(capacity: 1)
        let cfnum = CFNumberCreate(kCFAllocatorDefault, .intType, cfnumPointer)
        let keys: [CFString] = [kCVPixelBufferCGImageCompatibilityKey, kCVPixelBufferCGBitmapContextCompatibilityKey, kCVPixelBufferBytesPerRowAlignmentKey]
        let values: [CFTypeRef] = [kCFBooleanTrue, kCFBooleanTrue, cfnum!]
        let keysPointer = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
        let valuesPointer =  UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
        keysPointer.initialize(to: keys)
        valuesPointer.initialize(to: values)
        
        let options = CFDictionaryCreate(kCFAllocatorDefault, keysPointer, valuesPointer, keys.count, nil, nil)
        
        let width = cgimage!.width
        let height = cgimage!.height
        
        var pxbuffer: CVPixelBuffer?
        // if pxbuffer = nil, you will get status = -6661
        _ = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                kCVPixelFormatType_32BGRA, options, &pxbuffer)
        _ = CVPixelBufferLockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0));
        
        let bufferAddress = CVPixelBufferGetBaseAddress(pxbuffer!);
        
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB();
        let bytesperrow = CVPixelBufferGetBytesPerRow(pxbuffer!)
        let context = CGContext(data: bufferAddress,
                                width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: bytesperrow,
                                space: rgbColorSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue);
        context?.concatenate(CGAffineTransform(rotationAngle: 0))
        // context?.concatenate(__CGAffineTransformMake( 1, 0, 0, -1, 0, CGFloat(height) )) //Flip Vertical
        //        context?.concatenate(__CGAffineTransformMake( -1.0, 0.0, 0.0, 1.0, CGFloat(width), 0.0)) //Flip Horizontal
        
        
        context?.draw(cgimage!, in: CGRect(x:0, y:0, width:CGFloat(width), height:CGFloat(height)));
        _ = CVPixelBufferUnlockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0));
        return pxbuffer!;
    }
    
    func resized(targetSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        self.draw(in: CGRect(origin: .zero, size: targetSize))
        let toConvertImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return toConvertImage
    }
    
    func grayData(
        normalization: TFLiteVisionInterpreter.NormalizationOptions = .none,
        isModelQuantized: Bool,
        dataType: Tensor.DataType = .float32) -> Data? {
        
        let width = Int(size.width)
        let height = Int(size.height)

        let pixels = UnsafeMutablePointer<UInt32>.allocate(capacity: width * height * MemoryLayout<UInt32>.size)
        defer {
            pixels.deallocate()
        }
        memset(pixels, 0, width * height * MemoryLayout<UInt32>.size)
        
        let bitmapInfo: CGBitmapInfo = [.byteOrder32Little, CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)]
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        let context = CGContext(data: pixels, width: width, height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: width * MemoryLayout<UInt32>.size,
                                space: colorSpace,
                                bitmapInfo: bitmapInfo.rawValue)

        context?.draw(cgImage!, in: CGRect(x: 0, y: 0, width: 28, height: 28))

        let imageBytes: [UInt8] = (0..<(height * width)).map { i in
            let pixel = pixels[i].toUInt8s()
            return UInt8(Float(pixel[0]) * 0.3 + Float(pixel[1]) * 0.59 + Float(pixel[2]) * 0.11)
        }
        
        switch dataType {
        case .uInt8:
            return Data(copyingBufferOf: imageBytes)
        case .float32:
            switch normalization {
            case .none:
                return Data(copyingBufferOf: imageBytes.map { Float($0) })
            case .scaled(from: let from, to: let to):
                return Data(copyingBufferOf: imageBytes.map { element -> Float in ((Float(element) * (1.0 / 255.0)) * (to - from)) + from })
            case .meanStd(mean: let mean, std: let std):
                var bytes = imageBytes.map { Float($0) } // normalization
                for i in 0 ..< width * height {
                    bytes[width * height * 0 + i] = (Float32(imageBytes[i * 1 + 0]) - mean[0]) / std[0] // Gray
                }
                return Data(copyingBufferOf: bytes)
            }
        default:
            fatalError("don't support the type: \(dataType)")
        }
    }
    
    // https://github.com/tensorflow/examples/blob/master/lite/examples/digit_classifier/ios/DigitClassifier/TFLiteExtensions.swift
    /// Returns the data representation of the image after scaling to the given `size` and converting
    /// to grayscale.
    ///
    /// - Parameters
    ///   - size: Size to scale the image to (i.e. image size used while training the model).
    /// - Returns: The scaled image as data or `nil` if the image could not be scaled.
    func resizedGrayCGImage(with size: CGSize) -> CGImage? {
        guard let cgImage = self.cgImage, cgImage.width > 0, cgImage.height > 0 else { return nil }

        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        let width = Int(size.width)
        guard let context = CGContext(
          data: nil,
          width: width,
          height: Int(size.height),
          bitsPerComponent: cgImage.bitsPerComponent,
          bytesPerRow: width * 1,
          space: CGColorSpaceCreateDeviceGray(),
          bitmapInfo: bitmapInfo.rawValue)
          else {
            return nil
        }
        context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        return context.makeImage()
    }
}

extension UIView {
    func uiImage(in rect: CGRect) -> UIImage {
        let renderer = UIGraphicsImageRenderer(bounds: rect)
        return renderer.image { rendererContext in
            layer.render(in: rendererContext.cgContext)
        }
    }
}

extension UInt32 {
    func toUInt8s() -> [UInt8] {
        var bigEndian = self.bigEndian
        let count = MemoryLayout<UInt32>.size
        let bytePtr = withUnsafePointer(to: &bigEndian) {
            $0.withMemoryRebound(to: UInt8.self, capacity: count) {
                UnsafeBufferPointer(start: $0, count: count)
            }
        }
        return Array(bytePtr)
    }
}
