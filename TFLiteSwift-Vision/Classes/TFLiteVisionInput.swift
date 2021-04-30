//
//  TFLiteVisionInput.swift
//  TFLiteSwift-Vision
//
//  Created by Doyoung Gwak on 2021/04/13.
//

import Foundation

public enum TFLiteVisionInput {
    case pixelBuffer(pixelBuffer: CVPixelBuffer, preprocessOptions: PreprocessOptions)
    case uiImage(uiImage: UIImage, preprocessOptions: PreprocessOptions)
    
    var pixelBuffer: CVPixelBuffer? {
        switch self {
        case .pixelBuffer(let pixelBuffer, _):
            return pixelBuffer
        case .uiImage(let uiImage, _):
            return uiImage.pixelBufferFromImage()
        }
    }
    
    var cropArea: PreprocessOptions.CropArea {
        switch self {
        case .pixelBuffer(_, let preprocessOptions):
            return preprocessOptions.cropArea
        case .uiImage(_, let preprocessOptions):
            return preprocessOptions.cropArea
        }
    }
    
    var imageSize: CGSize {
        switch self {
        case .pixelBuffer(let pixelBuffer, _):
            return pixelBuffer.size
        case .uiImage(let uiImage, _):
            return uiImage.size
        }
    }
    
    var targetSquare: CGRect {
        switch cropArea {
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
    
    func croppedPixelBuffer(with inputModelSize: CGSize) -> CVPixelBuffer? {
        guard let pixelBuffer = pixelBuffer else { return nil }
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)
        
        // Resize `targetSquare` of input image to `modelSize`.
        return pixelBuffer.resize(from: targetSquare, to: inputModelSize)
    }
}
