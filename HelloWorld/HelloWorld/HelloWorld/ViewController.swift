import UIKit

class ViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var resultView: UITextView!
    private lazy var module: TorchModule = {
//        let model = "Structured3D_optimized"
//        let model = "dlab_mobile"
        let model = "mobilenet_mobile"
        if let filePath = Bundle.main.path(forResource: model, ofType: "pt"),
            let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model file!")
        }
    }()

    private lazy var labels: [String] = {
        if let filePath = Bundle.main.path(forResource: "words", ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Can't find the text file!")
        }
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        let image = UIImage(named: "rgb_rawlight.png")!
        imageView.image = image
        let resizedImage = image.resized(to: CGSize(width: 640, height: 384))
        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }

        let modelNames = ["Structured3D_optimized", "dlab_mobile", "mobilenet_mobile"]

        for name in modelNames {
            let path = Bundle.main.path(forResource: name, ofType: "pt")!
            let model = TorchModule(fileAtPath: path)!
            // Warmup
            var result = model.predict(image: UnsafeMutableRawPointer(&pixelBuffer))
            NSLog("before \(name)")
            for _ in 1...100 {
                result = model.predict(image: UnsafeMutableRawPointer(&pixelBuffer))
            }
            NSLog("after \(name)")
            NSLog("\(result)")
        }
    }
}
