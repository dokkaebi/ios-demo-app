#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation TorchModule {
 @protected
  torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
  self = [super init];
  if (self) {
    try {
      _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
    } catch (const std::exception& exception) {
      NSLog(@"%s", exception.what());
      return nil;
    }
  }
  return self;
}

- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
  try {
      const int WIDTH = 640;
      const int HEIGHT = 384;
      const int OUT_WIDTH = 160;
      const int OUT_HEIGHT = 96;
    at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, HEIGHT, WIDTH}, at::kFloat);
    c10::InferenceMode guard;
      auto output = _impl.forward({tensor});
      auto outputDict = output.toGenericDict();
      auto outputTensor = outputDict.at("plane_center").toTensor().cpu();
    float* floatBuffer = outputTensor.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < OUT_WIDTH * OUT_HEIGHT * 3; i++) {
      [results addObject:@(floatBuffer[i])];
    }
    return [results copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}

@end

