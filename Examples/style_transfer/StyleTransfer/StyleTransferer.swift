// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TFLiteSwift_Vision
import UIKit

class StyleTransferer {

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var predictInterpreter: TFLiteVisionInterpreter
  private var transferInterpreter: TFLiteVisionInterpreter

  /// Dedicated DispatchQueue for TF Lite operations.
  private let tfLiteQueue: DispatchQueue

  // MARK: - Initialization

  /// Create a Style Transferer instance with a quantized Int8 model that runs inference on the CPU.
  static func newCPUStyleTransferer(
    completion: @escaping ((Result<StyleTransferer>) -> Void)
  ) -> () {
    return StyleTransferer.newInstance(transferModel: Constants.Int8.transferModel,
                                       predictModel: Constants.Int8.predictModel,
                                       useMetalDelegate: false,
                                       completion: completion)
  }

  static func newGPUStyleTransferer(
    completion: @escaping ((Result<StyleTransferer>) -> Void)
  ) -> () {
    return StyleTransferer.newInstance(transferModel: Constants.Float16.transferModel,
                                       predictModel: Constants.Float16.predictModel,
                                       useMetalDelegate: true,
                                       completion: completion)
  }

  /// Create a new Style Transferer instance.
  static func newInstance(transferModel: String,
                          predictModel: String,
                          useMetalDelegate: Bool,
                          completion: @escaping ((Result<StyleTransferer>) -> Void)) {
    // Create a dispatch queue to ensure all operations on the Intepreter will run serially.
    let tfLiteQueue = DispatchQueue(label: "org.tensorflow.examples.lite.style_transfer")
    
    let accelerator: TFLiteVisionInterpreter.Accelerator = useMetalDelegate ? .metal : .cpu
    let threadCount: Int = ProcessInfo.processInfo.processorCount >= 2 ? 2 : 1

    // Run initialization in background thread to avoid UI freeze.
    tfLiteQueue.async {
      
      let transferModelOptions = TFLiteVisionInterpreter.Options(
        modelName: transferModel,
        threadCount: threadCount,
        accelerator: accelerator,
        normalization: .scaled(from: 0.0, to: 1.0)
      )
      
      let predictModelOptions = TFLiteVisionInterpreter.Options(
        modelName: predictModel,
        threadCount: threadCount,
        accelerator: accelerator,
        normalization: .scaled(from: 0.0, to: 1.0)
      )

      do {
        // Create the `Interpreter`s.
        let predictInterpreter = try TFLiteVisionInterpreter(options: predictModelOptions)
        let transferInterpreter = try TFLiteVisionInterpreter(options: transferModelOptions)

        // Create an StyleTransferer instance and return.
        let styleTransferer = StyleTransferer(
          tfLiteQueue: tfLiteQueue,
          predictInterpreter: predictInterpreter,
          transferInterpreter: transferInterpreter
        )
        DispatchQueue.main.async {
          completion(.success(styleTransferer))
        }
      } catch let error {
        print("Failed to create the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(InitializationError.internalError(error)))
        }
        return
      }
    }
  }

  /// Initialize Style Transferer instance.
  fileprivate init(
    tfLiteQueue: DispatchQueue,
    predictInterpreter: TFLiteVisionInterpreter,
    transferInterpreter: TFLiteVisionInterpreter
  ) {
    // Store TF Lite intepreter
    self.predictInterpreter = predictInterpreter
    self.transferInterpreter = transferInterpreter

    // Store the dedicated DispatchQueue for TFLite.
    self.tfLiteQueue = tfLiteQueue
  }

  // MARK: - Style Transfer

  /// Run style transfer on a given image.
  /// - Parameters
  ///   - styleImage: the image to use as a style reference.
  ///   - image: the target image.
  ///   - completion: the callback to receive the style transfer result.
  func runStyleTransfer(style styleImage: UIImage,
                        image: UIImage,
                        completion: @escaping ((Result<StyleTransferResult>) -> Void)) {
    tfLiteQueue.async {
      let startTime: Date = Date()
      var preprocessingTime: TimeInterval = 0
      var stylePredictTime: TimeInterval = 0
      var styleTransferTime: TimeInterval = 0
      var postprocessingTime: TimeInterval = 0

      func timeSinceStart() -> TimeInterval {
        return abs(startTime.timeIntervalSinceNow)
      }
      
      let output: TFLiteFlatArray<Float32>
      do {
        preprocessingTime = timeSinceStart()
        guard let predictOutput = try self.predictInterpreter.inference(with: styleImage).first else {
          return
        }
        
        let imageData = try self.transferInterpreter.preprocess(with: image)
        let bottleneckData = self.transferInterpreter.convertToData(with: predictOutput)
        stylePredictTime = timeSinceStart() - preprocessingTime
        
        output = try self.transferInterpreter.inference(with: [imageData, bottleneckData]).first!
        styleTransferTime = timeSinceStart() - stylePredictTime - preprocessingTime

      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(StyleTransferError.internalError(error)))
        }
        return
      }

      // Construct image from output tensor data
      let outputData = self.transferInterpreter.convertToData(with: output)
      guard let inputWidth = self.transferInterpreter.inputWidth, let inputHeight = self.transferInterpreter.inputHeight else { return }
      let size = CGSize(width: inputWidth, height: inputHeight)
      guard let cgImage = self.postprocessImageData(data: outputData, size: size) else {
        DispatchQueue.main.async {
          completion(.error(StyleTransferError.resultVisualizationError))
        }
        return
      }

      let outputImage = UIImage(cgImage: cgImage)

      postprocessingTime =
          timeSinceStart() - stylePredictTime - styleTransferTime - preprocessingTime

      // Return the result.
      DispatchQueue.main.async {
        completion(
          .success(
            StyleTransferResult(
              resultImage: outputImage,
              preprocessingTime: preprocessingTime,
              stylePredictTime: stylePredictTime,
              styleTransferTime: styleTransferTime,
              postprocessingTime: postprocessingTime
            )
          )
        )
      }
    }
  }

    // MARK: - Utils

  /// Turns TF model's float32 array output into one supported by `CGImage`. This method
  /// assumes the provided data is the same format as the data returned from the output
  /// tensor in `runStyleTransfer`, so it should not be used for general image processing.
  /// - Parameter data: The image data to turn into a `CGImage`. This data must be a buffer of
  ///   `Float32` values between 0 and 1 in RGB format.
  /// - Parameter size: The expected size of the output image.
  private func postprocessImageData(data: Data,
                                    size: CGSize) -> CGImage? {
    let width = Int(size.width)
    let height = Int(size.height)

    let floats = data.toArray(type: Float32.self)

    let bufferCapacity = width * height * 4
    let unsafePointer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferCapacity)
    let unsafeBuffer = UnsafeMutableBufferPointer<UInt8>(start: unsafePointer,
                                                         count: bufferCapacity)
    defer {
      unsafePointer.deallocate()
    }

    for x in 0 ..< width {
      for y in 0 ..< height {
        let floatIndex = (y * width + x) * 3
        let index = (y * width + x) * 4
        let red = UInt8(floats[floatIndex] * 255)
        let green = UInt8(floats[floatIndex + 1] * 255)
        let blue = UInt8(floats[floatIndex + 2] * 255)

        unsafeBuffer[index] = red
        unsafeBuffer[index + 1] = green
        unsafeBuffer[index + 2] = blue
        unsafeBuffer[index + 3] = 0
      }
    }

    let outData = Data(buffer: unsafeBuffer)

    // Construct image from output tensor data
    let alphaInfo = CGImageAlphaInfo.noneSkipLast
    let bitmapInfo = CGBitmapInfo(rawValue: alphaInfo.rawValue)
        .union(.byteOrder32Big)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard
      let imageDataProvider = CGDataProvider(data: outData as CFData),
      let cgImage = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: MemoryLayout<UInt8>.size * 4 * Int(size.width),
        space: colorSpace,
        bitmapInfo: bitmapInfo,
        provider: imageDataProvider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
      )
      else {
        return nil
    }
    return cgImage
  }

}

// MARK: - Types

/// Representation of the style transfer result.
struct StyleTransferResult {

  /// The resulting image from the style transfer.
  let resultImage: UIImage

  /// Time required to resize the input and style images and convert the image
  /// data to a format the model can accept.
  let preprocessingTime: TimeInterval

  /// The style prediction model run time.
  let stylePredictTime: TimeInterval

  /// The style transfer model run time.
  let styleTransferTime: TimeInterval

  /// Time required to convert the model output data to a `CGImage`.
  let postprocessingTime: TimeInterval

}

/// Convenient enum to return result with a callback
enum Result<T> {
  case success(T)
  case error(Error)
}

/// Define errors that could happen in the initialization of this class
enum InitializationError: Error {
  // Invalid TF Lite model
  case invalidModel(String)

  // Invalid label list
  case invalidLabelList(String)

  // TF Lite Internal Error when initializing
  case internalError(Error)
}

/// Define errors that could happen when running style transfer
enum StyleTransferError: Error {
  // Invalid input image
  case invalidImage

  // TF Lite Internal Error when initializing
  case internalError(Error)

  // Invalid input image
  case resultVisualizationError
}

// MARK: - Constants
private enum Constants {

  // Namespace for quantized Int8 models.
  enum Int8 {

    static let predictModel = "style_predict_quantized_256"

    static let transferModel = "style_transfer_quantized_384"

  }

  // Namespace for Float16 models, optimized for GPU inference.
  enum Float16 {

    static let predictModel = "style_predict_f16_256"

    static let transferModel = "style_transfer_f16_384"

  }

  static let modelFileExtension = "tflite"

}
