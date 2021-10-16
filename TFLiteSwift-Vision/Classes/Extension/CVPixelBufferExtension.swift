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
//  CVPixelBufferExtension.swift
//  PoseEstimation-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/03/16.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import Accelerate
import Foundation
import TensorFlowLite

extension CVPixelBuffer {
    var size: CGSize {
        return CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))
    }
    
    /// Returns a new `CVPixelBuffer` created by taking the self area and resizing it to the
    /// specified target size. Aspect ratios of source image and destination image are expected to be
    /// same.
    ///
    /// - Parameters:
    ///   - from: Source area of image to be cropped and resized.
    ///   - to: Size to scale the image to(i.e. image size used while training the model).
    /// - Returns: The cropped and resized image of itself.
    func resized(from source: CGRect, to size: CGSize) -> CVPixelBuffer? {
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
        let imageChannels = 4
        
        CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0)) }
        
        // Finds the address of the upper leftmost pixel of the source area.
        guard let baseAddress = CVPixelBufferGetBaseAddress(self)?
                .advanced(by: Int(source.minY) * inputImageRowBytes + Int(source.minX) * imageChannels) else {
            return nil
        }
      
        var sourceImage = vImage_Buffer(
            data: baseAddress,
            height: UInt(source.height),
            width: UInt(source.width),
            rowBytes: inputImageRowBytes
        )
        
        
        let resultRowBytes = Int(size.width) * imageChannels
        guard let baseAddress = malloc(Int(size.height) * resultRowBytes) else {
            return nil
        }
        
        // Allocates a vacant vImage buffer for resized image.
        var distinationImage = vImage_Buffer(
            data: baseAddress,
            height: UInt(size.height), width: UInt(size.width),
            rowBytes: resultRowBytes
        )
        
        guard vImageScale_ARGB8888(&sourceImage, &distinationImage, nil, vImage_Flags(0)) == kvImageNoError else {
            return nil
        }
        
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = { mutablePointer, pointer in
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }
        
        var result: CVPixelBuffer?
        
        // Converts the thumbnail vImage buffer to CVPixelBuffer
        let conversionStatus = CVPixelBufferCreateWithBytes(
            nil,
            Int(size.width), Int(size.height),
            CVPixelBufferGetPixelFormatType(self),
            baseAddress,
            resultRowBytes,
            releaseCallBack,
            nil,
            nil,
            &result
        )
        
        guard conversionStatus == kCVReturnSuccess else {
            free(baseAddress)
            return nil
        }
        
        return result
    }
    
    func rgbData(
        normalization: TFLiteVisionInterpreter.NormalizationOptions = .none,
        dataType: Tensor.DataType = .float32) -> Data? {
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
        guard let sourceData = CVPixelBufferGetBaseAddress(self) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(self)
        let destinationBytesPerRow = 3 * width
        
        // Assign input image to `sourceBuffer` to convert it.
        var sourceBuffer = vImage_Buffer(
            data: sourceData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: sourceBytesPerRow
        )
        
        // Make `destinationBuffer` and `destinationData` for its data to be assigned.
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            os_log("Error: out of memory", type: .error)
            return nil
        }
        defer { free(destinationData) }
        var destinationBuffer = vImage_Buffer(
            data: destinationData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: destinationBytesPerRow)
        
        // Convert image type.
        switch CVPixelBufferGetPixelFormatType(self) {
        case kCVPixelFormatType_32BGRA:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32ARGB:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32RGBA:
            vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        default:
            os_log("The type of this image is not supported.", type: .error)
            return nil
        }
        
        // Make `Data` with converted image.
        let imageByteData = Data(
            bytes: destinationBuffer.data,
            count: destinationBuffer.rowBytes * height
        )
        
        if dataType == .uInt8 { return imageByteData }
        
        let imageBytes = [UInt8](imageByteData)
        
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
                    bytes[width * height * 0 + i] = (Float32(imageBytes[i * 3 + 0]) - mean[0]) / std[0] // R
                    bytes[width * height * 1 + i] = (Float32(imageBytes[i * 3 + 1]) - mean[1]) / std[1] // G
                    bytes[width * height * 2 + i] = (Float32(imageBytes[i * 3 + 2]) - mean[2]) / std[2] // B
                }
                return Data(copyingBufferOf: bytes)
            }
        default:
            fatalError("don't support the type: \(dataType)")
        }
    }
    
    func grayData(
        normalization: TFLiteVisionInterpreter.NormalizationOptions = .none,
        isModelQuantized: Bool,
        dataType: Tensor.DataType = .float32) -> Data? {
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(self) else { return nil }
        
        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)
        let _ = CVPixelBufferGetBytesPerRow(self)
        
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        var imageBytes: [UInt8] = [UInt8](repeating: 0, count: width * height)
        imageBytes = imageBytes.enumerated().map { buffer[$0.offset] }
        
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
}
