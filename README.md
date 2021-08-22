# TFLiteSwift-Vision

[![CI Status](https://img.shields.io/travis/tucan9389/TFLiteSwift-Vision.svg?style=flat)](https://travis-ci.org/tucan9389/TFLiteSwift-Vision)
[![Version](https://img.shields.io/cocoapods/v/TFLiteSwift-Vision.svg?style=flat)](https://cocoapods.org/pods/TFLiteSwift-Vision)
[![License](https://img.shields.io/cocoapods/l/TFLiteSwift-Vision.svg?style=flat)](https://cocoapods.org/pods/TFLiteSwift-Vision)
[![Platform](https://img.shields.io/cocoapods/p/TFLiteSwift-Vision.svg?style=flat)](https://cocoapods.org/pods/TFLiteSwift-Vision)

## Goal

This libarary is a layer for vision's preprocessing and postprocessing when you are using [TensorFlowLiteSwift](https://cocoapods.org/pods/TensorFlowLiteSwift). You can use TFLiteSwift-Vision, if you want to implemented preprocessing and postprocessing functions in the repository. 

## Requirements

- [Xcode](https://developer.apple.com/xcode/)
- [CocodaPods](https://cocoapods.org/)

## Getting Started with Example

If you want to run the example of this library, you can do by executing following command line:

```shell
git clone https://github.com/tucan9389/TFLiteSwift-Vision
cd TFLiteSwift-Vision/Example
pod install
open TFLiteSwift-Vision.xcworkspace
```

## Installation in Your Own Project

TFLiteSwift-Vision is available through [CocoaPods](https://cocoapods.org). To install
it, simply add the following line to your Podfile:

```ruby
target 'MyXcodeProject' do
  use_frameworks!

  # Pods for Your Project
  pod 'TFLiteSwift-Vision'

end
```

## Done and TODO

TFLiteSwift-Vision is supporting (or wants to support) follow functions:

- Preprocessing (convert Cocoa's image type to TFLiteSwift's Tensor)

  - Supporting Cocoa image type:

    - [x] UIImage → Data

    - [x] CVPixelBuffer → Data
    - [ ] CGImage → Data

  - Supporting normalization methods:

    - [x] Normalization with scaling (0...255 → 0.0...1.0)
    - [x] Normalization with mean and std (normaly used in pytorch and it is used in ImageNet firstly)
    - [ ] Grayscaling (not to 4 dim tensor, but to 3 dim tensor from an image)

  - Supporting cropping methods:

    - [x] Resizing (`vImageScale_ARGB8888`)
    - [ ] Centercropping
    - [ ] Padding
    - If basic functions are implemented, need to optimize with Metal or Accelerate (or other domain specific frameworks)

- Inferenceing

  - batch size
    - [x] 1 batch
    - [ ] n batch
  - [x] cpu or gpu(metal) selectable

- Postprocessing (convert TFLiteSwift's Tensor to Cocoa's type)

  - [ ] Tensor to UIImage

## Author

tucan9389, tucan.dev@gmail.com

## License

TFLiteSwift-Vision is available under the Apache license. See the [LICENSE](LICENSE) file for more info.