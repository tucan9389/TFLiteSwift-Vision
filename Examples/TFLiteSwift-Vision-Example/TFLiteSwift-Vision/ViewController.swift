//
//  ViewController.swift
//  TFLiteSwift-Vision
//
//  Created by tucan9389 on 04/10/2021.
//  Copyright (c) 2021 tucan9389. All rights reserved.
//

import UIKit
import TFLiteSwift_Vision

class ViewController: UIViewController, UINavigationControllerDelegate {
    
    let picker = UIImagePickerController()
    
    var visionInterpreter: TFLiteVisionInterpreter?
    var labels: [String]?

    @IBOutlet weak var mainImageView: UIImageView!
    @IBOutlet weak var predictedCategoryLabel: UILabel!
    @IBOutlet weak var predictedIndexLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let interpreterOptions = TFLiteVisionInterpreter.Options(
            modelName: "mobilenet_v2_1.0_224",
            inputRankType: .bwhc,
            normalization: .scaled(from: 0.0, to: 1.0)
        )
        visionInterpreter = try? TFLiteVisionInterpreter(options: interpreterOptions)
        
         if let labelFilePath = Bundle.main.path(forResource: "labels_mobilenet_quant_v1_224", ofType: "txt") {
            labels = try? String(contentsOfFile: labelFilePath).split(separator: "\n").map { String($0) }
            // labels?.remove(at: 0)
        }
        
        print(labels ?? "N/A labels")
        
        picker.delegate = self
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction func pickImage(_ sender: Any) {
        present(picker, animated: true)
    }
}

extension ViewController: UIImagePickerControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        
        predictedCategoryLabel.text = "predicted label: predicting..."
        predictedIndexLabel.text = "predicted index: predicting..."
        
        if let uiImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
            
            DispatchQueue(label: "com.tucan9389.inference", qos: .userInteractive).async { [weak self] in
                guard let self = self else { return }
                
                // inference
                guard let output: TFLiteFlatArray<Float32> = self.visionInterpreter?.inference(with: uiImage)?.first
                    else { fatalError("Cannot inference") }
                 
                print(output.dimensions)
                let predictedIndex: Int = Int(output.argmax())
                
                guard let predictedLabel = self.labels?[predictedIndex]
                    else { fatalError("Cannot get label") }
                
                print("predicted: \(predictedIndex), \(predictedLabel)")
                DispatchQueue.main.async {
                    self.predictedCategoryLabel.text = "predicted label: \(predictedLabel)"
                    self.predictedIndexLabel.text    = "predicted index: \(predictedIndex)"
                }
            }
            
            mainImageView.image = uiImage
            
        } else {
            fatalError("Cannot load image from data")
        }
        
        picker.dismiss(animated: true)
    }
}

import Accelerate

// Postprocessing for classification output
extension TFLiteFlatArray where Element == Float32 {
    func argmax() -> UInt {
        let stride = vDSP_Stride(1)
        let n = vDSP_Length(array.count)
        var c: Float = .nan
        var i: vDSP_Length = 0
        vDSP_maxvi(array,
                   stride,
                   &c,
                   &i,
                   n)
        return i
    }
}
