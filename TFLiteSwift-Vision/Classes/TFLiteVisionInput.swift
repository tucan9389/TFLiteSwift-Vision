//
//  TFLiteVisionInput.swift
//  TFLiteSwift-Vision
//
//  Created by Doyoung Gwak on 2021/04/13.
//

import Foundation

public enum TFLiteVisionInput {
    case pixelBuffer(pixelBuffer: CVPixelBuffer)
    case uiImage(uiImage: UIImage)
    
    var pixelBuffer: CVPixelBuffer? {
        switch self {
        case .pixelBuffer(let pixelBuffer):
            return pixelBuffer
        case .uiImage(let uiImage):
            return uiImage.pixelBufferFromImage()
        }
    }
    var uiImage: UIImage? {
        switch self {
        case .pixelBuffer(_):
            return nil
        case .uiImage(let uiImage):
            return uiImage
        }
    }
    
    var imageSize: CGSize {
        switch self {
        case .pixelBuffer(let pixelBuffer):
            return pixelBuffer.size
        case .uiImage(let uiImage):
            return uiImage.size
        }
    }
    
    func targetSize(cropType: TFLiteVisionInterpreter.CropType) -> CGRect {
        switch cropType {
        case .customAspectFill(let rect):
            return rect
        case .squareAspectFill:
            let size = imageSize
            let minLength = min(size.width, size.height)
            return CGRect(x: (size.width - minLength) / 2,
                          y: (size.height - minLength) / 2,
                          width: minLength, height: minLength)
        }
    }
    
    func croppedPixelBuffer(with inputModelSize: CGSize, and cropType: TFLiteVisionInterpreter.CropType) -> CVPixelBuffer? {
        guard let pixelBuffer = pixelBuffer else { return nil }
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)
        
        let targetSize = targetSize(cropType: cropType)
        // Resize `targetSize` of input image to `modelSize`.
        return pixelBuffer.resize(from: targetSize, to: inputModelSize)
    }
    
    func resizedUIImage(with inputModelSize: CGSize) -> UIImage? {
        guard let uiImage = uiImage else { return nil }
        return uiImage.resized(targetSize: inputModelSize)
    }
}
