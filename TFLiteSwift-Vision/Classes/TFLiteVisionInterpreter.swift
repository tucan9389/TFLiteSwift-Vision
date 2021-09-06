
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

struct TFLiteResult {
    let outputTensors: [Tensor]
}

public class TFLiteVisionInterpreter {
    let interpreter: Interpreter
    let options: Options
    var inputTensor: Tensor?
    var outputTensors: [Tensor] = []
    
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
        }
    }
    
    public init(options: Options) {
        guard let modelPath = Bundle.main.path(forResource: options.modelName, ofType: "tflite") else {
            fatalError("Failed to load the model file with name: \(options.modelName).")
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
            fatalError("Failed to craete interpreter")
        }
        
        self.interpreter = interpreter
        self.options = options
        
        do {
            try setupTensor(with: interpreter, options: options)
        } catch {
            fatalError("Failed to setup tensor: \(error.localizedDescription)")
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
    
    public func preprocess(with input: TFLiteVisionInput) -> Data? {
        guard let inputWidth = inputWidth, let inputHeight = inputHeight else { return nil }
        
        let modelInputSize = CGSize(width: inputWidth, height: inputHeight)
        guard let thumbnail = input.croppedPixelBuffer(with: modelInputSize, and: options.cropType) else { return nil }
        
        // Remove the alpha component from the image buffer to get the initialized `Data`.
        let byteCount = 1 * inputHeight * inputWidth * options.inputChannel
        let inputDataType = inputTensor?.dataType ?? .float32
        guard let inputData = thumbnail.rgbData(byteCount: byteCount,
                                                normalization: options.normalization,
                                                isModelQuantized: options.isQuantized,
                                                dataType: inputDataType) else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }
        
        return inputData
    }
    
    public func preprocess(with pixelBuffer: CVPixelBuffer, from targetSquare: CGRect) -> Data? {
        guard let inputWidth = inputWidth, let inputHeight = inputHeight else { return nil }
        
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)
        
        // Resize `targetSquare` of input image to `modelSize`.
        let modelSize = CGSize(width: inputWidth, height: inputHeight)
        guard let thumbnail = pixelBuffer.resize(from: targetSquare, to: modelSize) else { return nil }
        
        // Remove the alpha component from the image buffer to get the initialized `Data`.
        let byteCount = 1 * inputHeight * inputWidth * options.inputChannel
        guard let inputData = thumbnail.rgbData(byteCount: byteCount,
                                                normalization: options.normalization,
                                                isModelQuantized: options.isQuantized) else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }
        
        return inputData
    }
    
    public func inference(with inputData: Data) -> [TFLiteFlatArray<Float32>]? {
        // Copy the initialized `Data` to the input `Tensor`.
        do {
            // Copy input into interpreter's 0th `Tensor`.
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            try interpreter.invoke()
            
            // Get the output `Tensor` to process the inference results.
            for (index) in 0..<outputTensors.count {
                outputTensors[index] = try interpreter.output(at: index)
            }
        } catch /*let error*/ {
            fatalError("Failed to invoke the interpreter with error:" + error.localizedDescription)
        }
        
        return outputTensors.map { TFLiteFlatArray(tensor: $0) }
    }
    
    public func inference(with uiImage: UIImage) -> [TFLiteFlatArray<Float32>]? {
        let input: TFLiteVisionInput = .uiImage(uiImage: uiImage)
        
        // preprocess
        guard let inputData: Data = preprocess(with: input)
            else { fatalError("Cannot preprcess") }
        
        // inference
        guard let outputs: [TFLiteFlatArray<Float32>] = inference(with: inputData)
            else { fatalError("Cannot inference") }
        
        return outputs
    }
    
    public func inference(with pixelBuffer: CVPixelBuffer) -> [TFLiteFlatArray<Float32>]? {
        let input: TFLiteVisionInput = .pixelBuffer(pixelBuffer: pixelBuffer)
        
        // preprocess
        guard let inputData: Data = preprocess(with: input)
            else { fatalError("Cannot preprcess") }
        
        // inference
        guard let outputs: [TFLiteFlatArray<Float32>] = inference(with: inputData)
            else { fatalError("Cannot inference") }
        
        return outputs
    }
}

extension TFLiteVisionInterpreter {
    public enum NormalizationOptions {
        case none                   // 0...255
        case scaled(from: Float, to: Float)
        case meanStd(mean: [Float], std: [Float])
        
        public static var pytorchNormalization: NormalizationOptions {
            return .meanStd(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
        }
    }
    
    public enum RankType {
        case bhwc // usually tensorflow model
        case bwhc
        case bcwh // usually pytorch model
        case bchw
    }
    
    public enum CropType {
        case customAspectFill(rect: CGRect)
        case squareAspectFill
    }
    
    public struct Options {
        let modelName: String
        let threadCount: Int
        let accelerator: Accelerator
        let isQuantized: Bool
        
        let inputRankType: RankType
        let isGrayScale: Bool
        var inputChannel: Int { return isGrayScale ? 1 : 3 }
        
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
            self.isQuantized = isQuantized
            self.inputRankType = inputRankType
            self.isGrayScale = isGrayScale
            self.normalization = normalization
            self.cropType = cropType
        }
    }
    
}

extension TFLiteVisionInterpreter {
    public enum Accelerator {
        case cpu
        case metal
    }
}

extension TFLiteVisionInterpreter {
    public struct ModelInput {
        static let batchSize = 1
        static let channel = 3 // rgb: 3, gray: 1
    }
}
