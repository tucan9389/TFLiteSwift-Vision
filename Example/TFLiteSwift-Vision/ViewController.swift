//
//  ViewController.swift
//  TFLiteSwift-Vision
//
//  Created by tucan9389 on 04/10/2021.
//  Copyright (c) 2021 tucan9389. All rights reserved.
//

import UIKit
import TFLiteSwift_Vision

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        let interpreterOptions = TFLiteVisionInterpreter.Options(modelName: "mobilenet_v2_1.0_224", inputWidth: 224, inputHeight: 224)
        let visionInterpreter = TFLiteVisionInterpreter(options: interpreterOptions)
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

}

