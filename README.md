# TFLiteSwift-Vision

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/tucan9389/TFLiteSwift-Vision/compare)
[![Version](https://img.shields.io/cocoapods/v/TFLiteSwift-Vision.svg?style=flat)](https://cocoapods.org/pods/TFLiteSwift-Vision)
[![License](https://img.shields.io/cocoapods/l/TFLiteSwift-Vision.svg?style=flat)](https://cocoapods.org/pods/TFLiteSwift-Vision)
[![Platform](https://img.shields.io/cocoapods/p/TFLiteSwift-Vision.svg?style=flat)](https://cocoapods.org/pods/TFLiteSwift-Vision)
<!-- [![CI Status](https://img.shields.io/travis/tucan9389/TFLiteSwift-Vision.svg?style=flat)](https://travis-ci.org/tucan9389/TFLiteSwift-Vision) -->



## Table

- [Goal](#goal)
- [Requirements](#Requirements)
- [Usage](#Usage)
- [Getting Started with Example](#Getting-Started-with-Example)
- [Architecture](#Architecture)
- [Done and TODO](#Done-and-TODO)
- [License](#License)

## Goal

This library is a [TensorFlowLiteSwift](https://cocoapods.org/pods/TensorFlowLiteSwift) wrapping library for vision pre/post-processing. You can use TFLiteSwift-Vision, if you want to implemented preprocessing and postprocessing functions in the repository. 

![image](https://user-images.githubusercontent.com/37643248/130391342-6b83f6a7-9748-401c-89d5-33148f3ec2cc.png)

[Here](https://github.com/tucan9389/TFLiteSwift-Vision/issues/3) is more detail of this repo's background and goal, and what is diff.

## Requirements

- [Xcode](https://developer.apple.com/xcode/)
- [CocodaPods](https://cocoapods.org/)
- iOS 10.0+

## Usage

### Install the TFLiteSwift-Vision

TFLiteSwift-Vision is available through [CocoaPods](https://cocoapods.org). To install
it, simply add the following line to your Podfile:

```ruby
target 'MyXcodeProject' do
  use_frameworks!

  # Pods for Your Project
  pod 'TFLiteSwift-Vision', '~> 0.2.6'

end

post_install do |installer|
  installer.pods_project.build_configurations.each do |config|
    config.build_settings["EXCLUDED_ARCHS[sdk=iphonesimulator*]"] = "arm64"
  end
end
```

And then, run following:

```shell
pod install
```

### Add `.tflite` file in the Xcode Project

<img width="500px" alt="importing-tflite-into-xcode" src="https://user-images.githubusercontent.com/37643248/130346788-19431b71-4ae6-47d2-9903-a90fb6a0c2d2.png">

> If you have other label file or other meta data file, also import it.

### Implement Inference Code

Import TFLiteSwift_Vision framework.

```swift
import TFLiteSwift_Vision
```

Setup interpreter.

```swift
let options = TFLiteVisionInterpreter.Options(
  modelName: "mobilenet_v2_1.0_224",
  inputRankType: .bwhc,
  normalization: .scaled(from: 0.0, to: 1.0)
)
var visionInterpreter = try? TFLiteVisionInterpreter(options: options)
```

Inference with an image. The following is an image classification case.

```swift
// inference
guard let output: TFLiteFlatArray<Float32> = try? self.visionInterpreter?.inference(with: uiImage)?.first
	else { fatalError("Cannot inference") }

// postprocess
let predictedIndex: Int = Int(output.argmax())
print("predicted index: \(predictedLabel)")
print(output.dimensions)
```


## Getting Started with Example

### Clone and open the Example project

```shell
git clone https://github.com/tucan9389/TFLiteSwift-Vision
cd TFLiteSwift-Vision/Example
pod install
open TFLiteSwift-Vision.xcworkspace
```

### Download model and label file

Download tflite model and label txt, and then import the files into Xcode project.

> You can also download the following files on [here](https://www.tensorflow.org/lite/guide/hosted_models)
- [mobilenet_v2_1.0_224.tflite](https://github.com/tucan9389/TFLiteSwift-Vision/releases/download/tflite-upload/mobilenet_v2_1.0_224.tflite)
- [labels_mobilenet_quant_v1_224.txt](https://github.com/tucan9389/TFLiteSwift-Vision/releases/download/tflite-upload/labels_mobilenet_quant_v1_224.txt)

### Build and Run

After build and run the project, you can test the model(`mobilenet_v2_1.0_224.tflite`) with your own image data.

| image classification |
| :-: |
| ![demo-tfliteswift-vision-example-001](https://user-images.githubusercontent.com/37643248/130346511-cfdb21ce-c22c-4aec-b1e6-c4da81ae94d5.gif) |


## Architecture

![tfliteswift-vision-architecture](https://user-images.githubusercontent.com/37643248/130388924-eab0313c-8b7a-422e-9877-a7e7d6c00448.png)

## Done and TODO

TFLiteSwift-Vision is supporting (or wants to support) follow functions:

- Preprocessing (convert Cocoa's image type to TFLiteSwift's Tensor)
  - Supporting input image type:
    - [x] UIImage
    - [x] CVPixelBuffer
    - [ ] CGImage
    - [ ] MTLTexture
  - Supporing bhwc sequence:
    - [x] bwhc (normally used in TF)
    - [x] bcwh (noramlly used in Pytorch)
    - [x] bwh (for gray image)
    - [x] bhw (for gray image)
    - [ ] whc
    - [ ] hwc
    - [ ] wh (for gray input)
  - Supporting normalization methods:
    - [x] Normalization with scaling (0...255 → 0.0...1.0)
    - [x] Normalization with mean and std (normaly used in pytorch and it is used in ImageNet firstly)
    - [x] Grayscaling (not to 4 dim tensor, but to 3 dim tensor from an image)
  - Supporting cropping methods:
    - [x] Resizing (`vImageScale_ARGB8888`)
    - [x] Centercropping
    - [ ] Padding
    - If basic functions are implemented, need to optimize with Metal or Accelerate (or other domain specific frameworks)
  - Supporting input type
    - [x] Float32
    - [x] UInt8
  - Supporting output type
    - [x] Float32
    - [ ] UInt8
- Inferenceing
  - batch size
    - [x] 1 batch
    - [ ] n batch
  - [x] cpu or gpu(metal) selectable
- Postprocessing (convert TFLiteSwift's Tensor to Cocoa's type)
  - [ ] Tensor → UIImage
  - [ ] Tensor → MTLTexture
- Domain specific postprocessing examples
  - [x] Image classification
  - [ ] Object detection
  - [ ] Semantic segmentation
  - [ ] Pose estimation
- Replace TensorFlowLiteSwift to TFLiteSwift-Vision in tensorflow/examples
  - [x] image_classification
  - [x] object_detection
  - [x] posenet
  - [x] digit_classification
  - [ ] semantic_segmentation


## Author

- [@tucan9389](https://github.com/tucan9389), tucan.dev@gmail.com
- [@Seonghun23](https://github.com/Seonghun23), kimsh777kr@gmail.com

## License

TFLiteSwift-Vision is available under the Apache license. See the [LICENSE](LICENSE) file for more info.
