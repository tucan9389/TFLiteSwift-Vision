#
# Be sure to run `pod lib lint TFLiteSwift-Vision.podspec' to ensure this is a
# valid spec before submitting.
#
# Any lines starting with a # are optional, but their use is encouraged
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html
#

Pod::Spec.new do |s|
  s.name             = 'TFLiteSwift-Vision'
  s.version          = '0.2.8'
  s.summary          = 'A layer for vision\'s pre/post-processing when you are using TensorFlowLiteSwift'

# This description is used to generate tags and improve search results.
#   * Think: What does it do? Why did you write it? What is the focus?
#   * Try to keep it short, snappy and to the point.
#   * Write the description between the DESC delimiters below.
#   * Finally, don't worry about the indent, CocoaPods strips it!

  s.description      = <<-DESC
  This framework is a layer for vision's preprocessing and postprocessing when you are using TensorFlowLiteSwift. You can use TFLiteSwift-Vision, if you want to implemented preprocessing and postprocessing functions in the repository.
                       DESC

  s.homepage         = 'https://github.com/tucan9389/TFLiteSwift-Vision'
  # s.screenshots     = 'www.example.com/screenshots_1', 'www.example.com/screenshots_2'
  s.license          = { :type => 'Apache 2.0', :file => 'LICENSE' }
  s.author           = { 'tucan9389' => 'tucan.dev@gmail.com', 'Seonghun23' => 'kimsh777kr@gmail.com' }
  s.source           = { :git => 'https://github.com/tucan9389/TFLiteSwift-Vision.git', :tag => s.version.to_s }
  # s.social_media_url = 'https://twitter.com/<TWITTER_USERNAME>'

  s.ios.deployment_target = '10.0'
  s.swift_version = '5.0'

  s.source_files = 'TFLiteSwift-Vision/Classes/**/*'
  
  s.pod_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }
  s.user_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }
  
  # s.resource_bundles = {
  #   'TFLiteSwift-Vision' => ['TFLiteSwift-Vision/Assets/*.png']
  # }

  # s.public_header_files = 'Pod/Classes/**/*.h'
  # s.frameworks = 'UIKit', 'MapKit'
  s.static_framework = true
  s.dependency 'TensorFlowLiteSwift/CoreML', '~> 2.4.0'
end
