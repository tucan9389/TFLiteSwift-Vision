
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
//  TFLiteVisionInterpreter.swift
//  PoseEstimation-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/03/14.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import TensorFlowLite

enum TFLiteVisionInterpreterError: Error {
    case initModelPathError(modelName: String)
    case initInterpreterInitError
    case initModelSetupError(error: Error)
    case preprocessWidthHeightChannelNilError
    case preprocessConvertToDataError
    case preprocessResizeError
    case invalidInputChannal(channel: Int, shape: [Int])
}

extension TFLiteVisionInterpreterError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .initModelPathError(modelName: let modelName):
            return "Failed to load the model file with name: \(modelName)."
        case .initInterpreterInitError:
            return "Failed to craete interpreter"
        case .initModelSetupError(let error):
            return "Failed to setup tensor: \(error.localizedDescription)"
        case .preprocessWidthHeightChannelNilError:
            return "Failed to preprocess because width, height or channel are nil."
        case .preprocessConvertToDataError:
            return "Failed to convert the image buffer to data."
        case .preprocessResizeError:
            return "Failed to resize image"
        case .invalidInputChannal(let channel, let shape):
            return "Invalid channel: \(channel) of the shape(\(shape)"
        }
    }
}

struct TFLiteResult {
    let outputTensors: [Tensor]
}

public class TFLiteVisionInterpreter {
    let interpreter: Interpreter
    var options: Options
    var inputTensor: Tensor?
    public var outputTensors: [Tensor] = []
    
    public var inputWidth: Int? {
        switch options.inputRankType {
        case .bwhc:
            return inputTensor?.shape.dimensions[1]
        case .bchw:
            return inputTensor?.shape.dimensions[3]
        case .bhwc:
            return inputTensor?.shape.dimensions[2]
        case .bcwh:
            return inputTensor?.shape.dimensions[2]
        case .bhw:
            return inputTensor?.shape.dimensions[2]
        case .bwh:
            return inputTensor?.shape.dimensions[1]
        }
    }
    
    public var inputHeight: Int? {
        switch options.inputRankType {
        case .bwhc:
            return inputTensor?.shape.dimensions[2]
        case .bchw:
            return inputTensor?.shape.dimensions[2]
        case .bhwc:
            return inputTensor?.shape.dimensions[1]
        case .bcwh:
            return inputTensor?.shape.dimensions[3]
        case .bhw:
            return inputTensor?.shape.dimensions[1]
        case .bwh:
            return inputTensor?.shape.dimensions[2]
        }
    }
    
    public var inputChannel: Int? {
        switch options.inputRankType {
        case .bwhc:
            return inputTensor?.shape.dimensions[3]
        case .bchw:
            return inputTensor?.shape.dimensions[1]
        case .bhwc:
            return inputTensor?.shape.dimensions[3]
        case .bcwh:
            return inputTensor?.shape.dimensions[1]
        case .bhw, .bwh:
            return 1
        }
    }
    
    public var isGrayImage: Bool {
        return inputChannel == 1
    }
  
    public var isQuantized: Bool {
        return inputTensor?.dataType != .float32
    }
  
    public var outputDataType: Tensor.DataType {
        // outputTensors.first?.quantizationParameters
        return outputTensors.first?.dataType ?? .float32
    }
    
    public init(options: Options) throws {
        guard let modelPath = Bundle.main.path(forResource: options.modelName, ofType: "tflite") else {
            throw TFLiteVisionInterpreterError.initModelPathError(modelName: options.modelName)
        }
        
        // Specify the options for the `Interpreter`.
        var interpreterOptions = Interpreter.Options()
        interpreterOptions.threadCount = options.threadCount
        
        // Specify the delegates for the `Interpreter`.
        let delegates: [CoreMLDelegate]
        switch options.accelerator {
        case .metal:
            if let delegate = CoreMLDelegate() {
                delegates = [delegate]
            } else {
                delegates = []
            }
        default:
            delegates = []
        }
        
        guard let interpreter = try? Interpreter(modelPath: modelPath, options: interpreterOptions, delegates: delegates) else {
            throw TFLiteVisionInterpreterError.initInterpreterInitError
        }
        
        self.interpreter = interpreter
        self.options = options
        
        do {
            try setupTensor(with: interpreter, options: options)
        } catch {
            throw TFLiteVisionInterpreterError.initModelSetupError(error: error)
        }
      
        // Check options validation
        if options.inputRankType.rankCount == 4, inputTensor?.shape.dimensions.count == 3 {
            self.options.inputRankType = options.inputRankType.reducedRank()
        }
    }
    
    private func setupTensor(with interpreter: Interpreter, options: Options) throws {
        // Initialize input and output `Tensor`s.
        // Allocate memory for the model's input `Tensor`s.
        try interpreter.allocateTensors()
        
        // input tensor
        let inputTensor = try interpreter.input(at: 0)
        
        // input cases
        // 1. (bs, w, h, c)
        // 2. (bs, c, h, w)
        // 3. (w, h, c) color or gray <#TODO#>
        // 4. (c, h, w) color or gray <#TODO#>
        // 5. (w, h) gray <#TODO#>
        // 6. (h, w) gray <#TODO#>
        
        // print("inputTensor.dataType:", inputTensor.dataType)
        // float32, uInt8
        // maybe exist: float16, uInt4, ...,

        self.inputTensor = inputTensor
        
        print("------------------------------------------------------")
        print("inputTensor.shape.dimensions:", inputTensor.shape.dimensions)
        print("inputTensor.dataType:", inputTensor.dataType)
        print("------------------------------------------------------")
        
        try interpreter.invoke()
        
        // output tensor
        let outputTensors = try (0..<interpreter.outputTensorCount).map { outputTensorIndex -> Tensor in
            let outputTensor = try interpreter.output(at: outputTensorIndex)
            return outputTensor
        }
        
        self.outputTensors = outputTensors
    }
    
    public func preprocess(with input: TFLiteVisionInput, from targetSquare: CGRect? = nil) throws -> Data {
        guard let inputWidth = inputWidth, let inputHeight = inputHeight, let inputChannel = inputChannel else {
            throw TFLiteVisionInterpreterError.preprocessWidthHeightChannelNilError
        }
        
        if inputChannel == 3 {
            let modelInputSize = CGSize(width: inputWidth, height: inputHeight)
            guard let thumbnail = input.croppedPixelBuffer(with: modelInputSize, and: options.cropType, from: targetSquare) else {
                throw TFLiteVisionInterpreterError.preprocessResizeError
            }
            
            // Remove the alpha component from the image buffer to get the initialized `Data`.
            // let byteCount = 1 * inputHeight * inputWidth * inputChannel
            
            let inputDataType = inputTensor?.dataType ?? .float32
            guard let inputData = thumbnail.rgbData(normalization: options.normalization,
                                                    dataType: inputDataType)
            else {
                throw TFLiteVisionInterpreterError.preprocessConvertToDataError
            }
                                              
            return inputData
        } else if inputChannel == 1 {
            let modelInputSize = CGSize(width: inputWidth, height: inputHeight)
            guard let resizedCGImage = input.resizedGrayCGImage(with: modelInputSize) else {
                throw TFLiteVisionInterpreterError.preprocessResizeError
            }
            
            let inputDataType = inputTensor?.dataType ?? .float32
            guard let inputData = resizedCGImage.grayData(normalization: options.normalization,
                                                          dataType: inputDataType)
            else {
              throw TFLiteVisionInterpreterError.preprocessConvertToDataError
            }
                                              
            return inputData
        } else {
            throw TFLiteVisionInterpreterError.invalidInputChannal(channel: inputChannel, shape: [inputHeight, inputWidth, inputChannel])
        }
    }
    
    public func preprocess(with pixelBuffer: CVPixelBuffer, from targetSquare: CGRect) throws -> Data {
        guard let inputWidth = inputWidth, let inputHeight = inputHeight, let inputChannel = inputChannel else {
            throw TFLiteVisionInterpreterError.preprocessWidthHeightChannelNilError
        }
        
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)
        
        // Resize `targetSquare` of input image to `modelSize`.
        let modelSize = CGSize(width: inputWidth, height: inputHeight)
        guard let thumbnail = pixelBuffer.resized(from: targetSquare, to: modelSize) else {
            throw TFLiteVisionInterpreterError.preprocessResizeError
        }
        
        // Remove the alpha component from the image buffer to get the initialized `Data`.
        // let byteCount = 1 * inputHeight * inputWidth * inputChannel
        
        if inputChannel == 3 {
            guard let inputData = thumbnail.rgbData(normalization: options.normalization,
                                                    dataType: inputTensor?.dataType ?? .float32) else {
                throw TFLiteVisionInterpreterError.preprocessConvertToDataError
            }
          return inputData
        } else if inputChannel == 1 {
            guard let inputData = thumbnail.grayData(normalization: options.normalization,
                                                     isModelQuantized: isQuantized) else {
                throw TFLiteVisionInterpreterError.preprocessConvertToDataError
            }
          return inputData
        } else {
            throw TFLiteVisionInterpreterError.invalidInputChannal(channel: inputChannel, shape: [inputHeight, inputWidth, inputChannel])
        }
    }
  
    public func preprocess(with uiImage: UIImage) throws -> Data {
        let input: TFLiteVisionInput = .uiImage(uiImage: uiImage)
        
        // preprocess
        return try preprocess(with: input)
    }
  
    public func inference(with inputDataArray: [Data]) throws -> [TFLiteFlatArray] {
        // Copy input into interpreter's all input `Tensor`.
        do {
            try inputDataArray.enumerated().forEach { index, inputData in
                try interpreter.copy(inputData, toInputAt: index)
            }
        } catch let error {
            throw error
        }
      
        // Run inference by invoking the `Interpreter`.
        try interpreter.invoke()
      
        // Get the output `Tensor` to process the inference results.
        for (index) in 0..<outputTensors.count {
            outputTensors[index] = try interpreter.output(at: index)
        }
      
        return try outputTensors.map { try TFLiteFlatArray(tensor: $0) }
    }
    
    public func inference(with uiImage: UIImage) throws -> [TFLiteFlatArray] {
        let input: TFLiteVisionInput = .uiImage(uiImage: uiImage)
        
        // preprocess
        let inputData: Data = try preprocess(with: input)
        
        // inference
        let outputs: [TFLiteFlatArray] = try inference(with: [inputData])
        
        return outputs
    }
    
    public func inference(with pixelBuffer: CVPixelBuffer, from targetSquare: CGRect? = nil) throws -> [TFLiteFlatArray] {
        let input: TFLiteVisionInput = .pixelBuffer(pixelBuffer: pixelBuffer)
        
        // preprocess
        let inputData: Data = try preprocess(with: input, from: targetSquare)
        
        // inference
        let outputs: [TFLiteFlatArray] = try inference(with: [inputData])
        
        return outputs
    }
}

extension TFLiteVisionInterpreter {
    public enum NormalizationOptions {
        case none                   // 0...255
        case scaled(from: Float, to: Float)
        case meanStd(mean: [Float], std: [Float])
        
        public static var pytorchNormalization: NormalizationOptions {
            // https://github.com/jacobgil/pytorch-grad-cam/issues/6
            return .meanStd(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
        }
    }
    
    public enum RankType {
        case bhwc // usually tensorflow model
        case bwhc
        case bcwh // usually pytorch model
        case bchw
        case bhw  // only gray image
        case bwh  // only gray image
      
        public var rankCount: Int {
            switch self {
            case .bwhc, .bchw, .bhwc, .bcwh:
                return 4
            case .bhw, .bwh:
                return 3
            }
        }
      
        public func reducedRank() -> Self {
            switch self {
            case .bwhc:
                return .bwh
            case .bchw:
                return .bhw
            case .bhwc:
                return .bhw
            case .bcwh:
                return .bwh
            case .bhw:
                return .bhw
            case .bwh:
                return .bwh
            }
        }
    }
    
    public enum CropType {
        case customAspectFill(rect: CGRect)
        case squareAspectFill
        case scaleFill
    }
    
    public struct Options {
        let modelName: String
        let threadCount: Int
        let accelerator: Accelerator
        
        var inputRankType: RankType
        
        let normalization: NormalizationOptions
        let cropType: CropType
        
        public init(
            modelName: String,
            threadCount: Int = 1,
            accelerator: Accelerator = .cpu,
            isQuantized: Bool = false,
            inputRankType: RankType = .bwhc,
            isGrayScale: Bool = false,
            normalization: NormalizationOptions = .scaled(from: 0.0, to: 1.0),
            cropType: CropType = .squareAspectFill
        ) {
            self.modelName = modelName
            self.threadCount = threadCount
            #if targetEnvironment(simulator)
            self.accelerator = .cpu
            #else
            self.accelerator = accelerator
            #endif
            self.inputRankType = inputRankType
            self.normalization = normalization
            self.cropType = cropType
        }
    }
    
}

extension TFLiteVisionInterpreter {
    public enum Accelerator: Int, CaseIterable {
        case cpu
        case metal
      
        public var description: String {
            switch self {
            case .cpu:
                return "CPU"
            case .metal:
                return "GPU"
            }
        }
    }
}

extension TFLiteVisionInterpreter {
    public struct ModelInput {
        static let batchSize = 1
        static let channel = 3 // rgb: 3, gray: 1
    }
}
