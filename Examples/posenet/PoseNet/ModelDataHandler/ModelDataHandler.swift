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

import Accelerate
import CoreImage
import Foundation
import TFLiteSwift_Vision
import UIKit

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained.
class ModelDataHandler {
  // MARK: - Private Properties

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: TFLiteVisionInterpreter

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model is
  /// successfully loaded from the app's main bundle. Default `threadCount` is 2.
  init(
    threadCount: Int = Constants.defaultThreadCount,
    accelerator: TFLiteVisionInterpreter.Accelerator = .cpu
  ) throws {
    
    // Specify the options for the `TFLiteVisionInterpreter`.
    let options = TFLiteVisionInterpreter.Options(
      modelName: Model.file.name,
      threadCount: threadCount,
      accelerator: accelerator,
      normalization: .scaled(from: 0.0, to: 1.0),
      cropType: .scaleFill
    )
    
    // Create the `TFLiteVisionInterpreter`.
    interpreter = try TFLiteVisionInterpreter(options: options)
    
  }

  /// Runs PoseNet model with given image with given source area to destination area.
  ///
  /// - Parameters:
  ///   - on: Input image to run the model.
  ///   - from: Range of input image to run the model.
  ///   - to: Size of view to render the result.
  /// - Returns: Result of the inference and the times consumed in every steps.
  func runPoseNet(on pixelbuffer: CVPixelBuffer, from source: CGRect, to dest: CGSize)
    -> (Result, Times)?
  {
    // Start times of each process.
    let inferenceStartTime: Date
    let postprocessingStartTime: Date

    // Processing times in milliseconds.
    let inferenceTime: TimeInterval
    let postprocessingTime: TimeInterval

    inferenceStartTime = Date()
    guard let outputs = try? interpreter.inference(with: pixelbuffer, from: source) else {
      os_log("Inference failed", type: .error)
      return nil
    }
    inferenceTime = Date().timeIntervalSince(inferenceStartTime) * 1000

    postprocessingStartTime = Date()
    guard let result = postprocess(outputs: outputs, to: dest) else {
      os_log("Postprocessing failed", type: .error)
      return nil
    }
    postprocessingTime = Date().timeIntervalSince(postprocessingStartTime) * 1000

    let times = Times(
      inference: inferenceTime,
      postprocessing: postprocessingTime)
    return (result, times)
  }

  // MARK: - Private functions to run model
  /// Postprocesses output `Tensor`s to `Result` with size of view to render the result.
  ///
  /// - Parameters:
  ///   - to: Size of view to be displayed.
  /// - Returns: Postprocessed `Result`. `nil` if it can not be processed.
  private func postprocess(outputs: [TFLiteFlatArray<Float32>], to viewSize: CGSize) -> Result? {
    guard let inputWidth = interpreter.inputWidth, let inputHeight = interpreter.inputHeight
      else { return nil }
    
    // MARK: Formats output tensors
    // Convert `Tensor` to `FlatArray`. As PoseNet is not quantized, convert them to Float type
    // `FlatArray`.
    let heats = outputs[0]
    let offsets = outputs[1]

    // MARK: Find position of each key point
    // Finds the (row, col) locations of where the keypoints are most likely to be. The highest
    // `heats[0, row, col, keypoint]` value, the more likely `keypoint` being located in (`row`,
    // `col`).
    let keypointPositions = (0..<Model.output.keypointSize).map { keypoint -> (Int, Int) in
      var maxValue = heats[0, 0, 0, keypoint]
      var maxRow = 0
      var maxCol = 0
      for row in 0..<Model.output.height {
        for col in 0..<Model.output.width {
          if heats[0, row, col, keypoint] > maxValue {
            maxValue = heats[0, row, col, keypoint]
            maxRow = row
            maxCol = col
          }
        }
      }
      return (maxRow, maxCol)
    }

    // MARK: Calculates total confidence score
    // Calculates total confidence score of each key position.
    let totalScoreSum = keypointPositions.enumerated().reduce(0.0) { accumulator, elem -> Float32 in
      accumulator + sigmoid(heats[0, elem.element.0, elem.element.1, elem.offset])
    }
    let totalScore = totalScoreSum / Float32(Model.output.keypointSize)

    // MARK: Calculate key point position on model input
    // Calculates `KeyPoint` coordination model input image with `offsets` adjustment.
    let coords = keypointPositions.enumerated().map { index, elem -> (y: Float32, x: Float32) in
      let (y, x) = elem
      let yCoord =
        Float32(y) / Float32(Model.output.height - 1) * Float32(inputHeight)
        + offsets[0, y, x, index]
      let xCoord =
        Float32(x) / Float32(Model.output.width - 1) * Float32(inputWidth)
        + offsets[0, y, x, index + Model.output.keypointSize]
      return (y: yCoord, x: xCoord)
    }

    // MARK: Transform key point position and make lines
    // Make `Result` from `keypointPosition'. Each point is adjusted to `ViewSize` to be drawn.
    var result = Result(dots: [], lines: [], score: totalScore)
    var bodyPartToDotMap = [BodyPart: CGPoint]()
    for (index, part) in BodyPart.allCases.enumerated() {
      let position = CGPoint(
        x: CGFloat(coords[index].x) * viewSize.width / CGFloat(inputWidth),
        y: CGFloat(coords[index].y) * viewSize.height / CGFloat(inputHeight)
      )
      bodyPartToDotMap[part] = position
      result.dots.append(position)
    }

    do {
      try result.lines = BodyPart.lines.map { map throws -> Line in
        guard let from = bodyPartToDotMap[map.from] else {
          throw PostprocessError.missingBodyPart(of: map.from)
        }
        guard let to = bodyPartToDotMap[map.to] else {
          throw PostprocessError.missingBodyPart(of: map.to)
        }
        return Line(from: from, to: to)
      }
    } catch PostprocessError.missingBodyPart(let missingPart) {
      os_log("Postprocessing error: %s is missing.", type: .error, missingPart.rawValue)
      return nil
    } catch {
      os_log("Postprocessing error: %s", type: .error, error.localizedDescription)
      return nil
    }

    return result
  }

  /// Returns value within [0,1].
  private func sigmoid(_ x: Float32) -> Float32 {
    return (1.0 / (1.0 + exp(-x)))
  }
}

// MARK: - Data types for inference result
struct KeyPoint {
  var bodyPart: BodyPart = BodyPart.NOSE
  var position: CGPoint = CGPoint()
  var score: Float = 0.0
}

struct Line {
  let from: CGPoint
  let to: CGPoint
}

struct Times {
  var inference: Double
  var postprocessing: Double
}

struct Result {
  var dots: [CGPoint]
  var lines: [Line]
  var score: Float
}

enum BodyPart: String, CaseIterable {
  case NOSE = "nose"
  case LEFT_EYE = "left eye"
  case RIGHT_EYE = "right eye"
  case LEFT_EAR = "left ear"
  case RIGHT_EAR = "right ear"
  case LEFT_SHOULDER = "left shoulder"
  case RIGHT_SHOULDER = "right shoulder"
  case LEFT_ELBOW = "left elbow"
  case RIGHT_ELBOW = "right elbow"
  case LEFT_WRIST = "left wrist"
  case RIGHT_WRIST = "right wrist"
  case LEFT_HIP = "left hip"
  case RIGHT_HIP = "right hip"
  case LEFT_KNEE = "left knee"
  case RIGHT_KNEE = "right knee"
  case LEFT_ANKLE = "left ankle"
  case RIGHT_ANKLE = "right ankle"

  /// List of lines connecting each part.
  static let lines = [
    (from: BodyPart.LEFT_WRIST, to: BodyPart.LEFT_ELBOW),
    (from: BodyPart.LEFT_ELBOW, to: BodyPart.LEFT_SHOULDER),
    (from: BodyPart.LEFT_SHOULDER, to: BodyPart.RIGHT_SHOULDER),
    (from: BodyPart.RIGHT_SHOULDER, to: BodyPart.RIGHT_ELBOW),
    (from: BodyPart.RIGHT_ELBOW, to: BodyPart.RIGHT_WRIST),
    (from: BodyPart.LEFT_SHOULDER, to: BodyPart.LEFT_HIP),
    (from: BodyPart.LEFT_HIP, to: BodyPart.RIGHT_HIP),
    (from: BodyPart.RIGHT_HIP, to: BodyPart.RIGHT_SHOULDER),
    (from: BodyPart.LEFT_HIP, to: BodyPart.LEFT_KNEE),
    (from: BodyPart.LEFT_KNEE, to: BodyPart.LEFT_ANKLE),
    (from: BodyPart.RIGHT_HIP, to: BodyPart.RIGHT_KNEE),
    (from: BodyPart.RIGHT_KNEE, to: BodyPart.RIGHT_ANKLE),
  ]
}

// MARK: - Custom Errors
enum PostprocessError: Error {
  case missingBodyPart(of: BodyPart)
}

// MARK: - Information about the model file.
typealias FileInfo = (name: String, extension: String)

enum Model {
  static let file: FileInfo = (
    name: "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped", extension: "tflite"
  )

  static let output = (batchSize: 1, height: 9, width: 9, keypointSize: 17, offsetSize: 34)
  static let isQuantized = false
}
