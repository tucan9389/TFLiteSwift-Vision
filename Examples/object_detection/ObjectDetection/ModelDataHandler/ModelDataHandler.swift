// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import CoreImage
import TFLiteSwift_Vision
import UIKit
import Accelerate

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
}

/// Stores one formatted inference.
struct Inference {
  let confidence: Float
  let className: String
  let rect: CGRect
  let displayColor: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet SSD model.
enum MobileNetSSD {
  static let modelInfo: FileInfo = (name: "detect", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labelmap", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {

  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int
  let threadCountLimit = 10

  let threshold: Float = 0.5

  // MARK: Model parameters
  let batchSize = 1
  var inputChannels: Int { interpreter.inputChannel ?? -1 }
  var inputWidth: Int { interpreter.inputWidth ?? -1 }
  var inputHeight: Int { interpreter.inputHeight ?? -1 }

  // image mean and std for floating model, should be consistent with parameters used in model training
  let imageMean: Float = 127.5
  let imageStd:  Float = 127.5

  // MARK: Private properties
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: TFLiteVisionInterpreter

  private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  private let colorStrideValue = 10
  private let colors = [
    UIColor.red,
    UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
    UIColor.green,
    UIColor.orange,
    UIColor.blue,
    UIColor.purple,
    UIColor.magenta,
    UIColor.yellow,
    UIColor.cyan,
    UIColor.brown
  ]

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    let options = TFLiteVisionInterpreter.Options(
      modelName: modelFilename,
      threadCount: threadCount,
      normalization: .meanStd(mean: [imageMean, imageMean, imageMean], std: [imageStd, imageStd, imageStd]),
      cropType: .scaleFill
    )
    do {
      // Create the `TFLiteVisionInterpreter` and allocate.
      interpreter = try TFLiteVisionInterpreter(options: options)
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    super.init()

    // Load the classes listed in the labels file.
    loadLabels(fileInfo: labelsFileInfo)
  }

  /// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
    
    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    
    let interval: TimeInterval
    
    let boundingBox: [Float]
    let outputClasses: [Float]
    let outputScores: [Float]
    let outputCount: Int
    
    do {
      let startDate = Date()
      let outputs = try interpreter.inference(with: pixelBuffer)
      interval = Date().timeIntervalSince(startDate) * 1000
      
      boundingBox = outputs[0].array
      outputClasses = outputs[1].array
      outputScores = outputs[2].array
      outputCount = Int(outputs[3][0])
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }
    
    // Formats the results
    let resultArray = formatResults(
      boundingBox: boundingBox,
      outputClasses: outputClasses,
      outputScores: outputScores,
      outputCount: outputCount,
      width: CGFloat(imageWidth),
      height: CGFloat(imageHeight)
    )

    // Returns the inference time and inferences
    let result = Result(inferenceTime: interval, inferences: resultArray)
    return result
  }

  /// Filters out all the results with confidence score < threshold and returns the top N results
  /// sorted in descending order.
  func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat) -> [Inference]{
    var resultsArray: [Inference] = []
    if (outputCount == 0) {
      return resultsArray
    }
    for i in 0...outputCount - 1 {

      let score = outputScores[i]

      // Filters results with confidence < threshold.
      guard score >= threshold else {
        continue
      }

      // Gets the output class names for detected classes from labels list.
      let outputClassIndex = Int(outputClasses[i])
      let outputClass = labels[outputClassIndex + 1]

      var rect: CGRect = CGRect.zero

      // Translates the detected bounding box to CGRect.
      rect.origin.y = CGFloat(boundingBox[4*i])
      rect.origin.x = CGFloat(boundingBox[4*i+1])
      rect.size.height = CGFloat(boundingBox[4*i+2]) - rect.origin.y
      rect.size.width = CGFloat(boundingBox[4*i+3]) - rect.origin.x

      // The detected corners are for model dimensions. So we scale the rect with respect to the
      // actual image dimensions.
      let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))

      // Gets the color assigned for the class
      let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
      let inference = Inference(confidence: score,
                                className: outputClass,
                                rect: newRect,
                                displayColor: colorToAssign)
      resultsArray.append(inference)
    }

    // Sort results in descending order of confidence.
    resultsArray.sort { (first, second) -> Bool in
      return first.confidence  > second.confidence
    }

    return resultsArray
  }

  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels(fileInfo: FileInfo) {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name " +
                   "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      labels = contents.components(separatedBy: .newlines)
    } catch {
      fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                   "valid labels file and try again.")
    }
  }

  /// This assigns color for a particular class.
  private func colorForClass(withIndex index: Int) -> UIColor {

    // We have a set of colors and the depending upon a stride, it assigns variations to of the base
    // colors to each object based on its index.
    let baseColor = colors[index % colors.count]

    var colorToAssign = baseColor

    let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)

    if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
      colorToAssign = modifiedColor
    }

    return colorToAssign
  }
}
