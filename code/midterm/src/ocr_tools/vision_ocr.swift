import CoreGraphics
import Darwin
import Foundation
import ImageIO
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: vision_ocr <image-path>")
}

let path = CommandLine.arguments[1]
let url = URL(fileURLWithPath: path)
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
    fail("could not open image: \(path)")
}
guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fail("could not decode image: \(path)")
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .fast
request.usesLanguageCorrection = true
request.recognitionLanguages = ["en-US"]

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fail("Vision OCR failed: \(error.localizedDescription)")
}

let observations = request.results ?? []
let ordered = observations.sorted { left, right in
    let leftY = left.boundingBox.origin.y
    let rightY = right.boundingBox.origin.y
    if abs(leftY - rightY) > 0.02 {
        return leftY > rightY
    }
    return left.boundingBox.origin.x < right.boundingBox.origin.x
}

for observation in ordered {
    if let candidate = observation.topCandidates(1).first {
        print(candidate.string)
    }
}
