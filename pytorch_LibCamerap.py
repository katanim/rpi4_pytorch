import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import list_models, get_model
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import ast

# Enable hardware optimizations
torch.backends.quantized.engine = 'qnnpack'
torch.set_num_threads(4)  # Adjust based on your Pi's cores
torch.set_num_interop_threads(1)  # Optimize memory usage

# Initialize camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"format": 'RGB888',
          "size": (224, 224)},
    controls={"FrameRate": 30}
)
picam2.configure(camera_config)
picam2.start()

# Load classes once, outside the loop
with open("mobilevnet_classes.txt", "r") as f:
    classes = ast.literal_eval(f.read())

# List available models
all_models = list_models()
classification_models = list_models(module=torchvision.models)

# Initialize a model - Change to quantized version
model = models.quantization.mobilenet_v3_large(pretrained=True, quantize=True)
model.eval()
model = torch.jit.script(model)
model = torch.jit.freeze(model)

start_time = time.time()
last_logged = time.time()
global_frame_count = 1

# Create window before the loop
cv2.namedWindow("Live Stream", cv2.WINDOW_AUTOSIZE)

with torch.no_grad():
    while True:
        try:
            # read frame
            image = picam2.capture_array()
            image = np.flip(image, axis=0).copy()
            # Convert RGB to BGR for OpenCV display
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Live Stream", bgr_image)
            
            # Process key events - Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Convert to RGB and normalize
            input_tensor = torch.from_numpy(image).float() / 255.0
            input_tensor = input_tensor.permute(2, 0, 1)
            input_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(input_tensor)
            input_batch = input_tensor.unsqueeze(0)

            # run model
            output = model(input_batch)

            # Process and display the top prediction
            probabilities = output[0].softmax(dim=0)
            idx = probabilities.argmax().item()
            confidence = probabilities[idx].item() * 100
            detected_object = classes[idx]

            if confidence > 25:  # Threshold for detection
                print(f"Detected: {detected_object} with confidence {confidence:.2f}% at frame {global_frame_count}")

            # log model performance
            global_frame_count += 1

        except KeyboardInterrupt:
            break

end_time = time.time()
elapsed_time = end_time - start_time
processing_fps = global_frame_count / elapsed_time
print("Processed {} frames in {:.2f} seconds (Processing FPS: {:.2f})".format(global_frame_count, elapsed_time, processing_fps))

picam2.stop()
cv2.destroyAllWindows()
