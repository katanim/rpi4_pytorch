import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import list_models, get_model
import cv2
from PIL import Image
import time
import ast


# Enable hardware optimizations
torch.backends.quantized.engine = 'qnnpack'
torch.set_num_threads(4)  # Adjust based on your Pi's cores
torch.set_num_interop_threads(1)  # Optimize memory usage

# Enable OpenCV optimizations
cv2.setNumThreads(4)  # Adjust based on your CPU cores
cv2.ocl.setUseOpenCL(True)

# Load classes
with open("mobilevnet_classes.txt", "r") as f:
    classes = ast.literal_eval(f.read())

# Open video file
video_path = "bolt-detection.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

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
frame_count = 0
global_frame_count = 1
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with torch.no_grad():
    while cap.isOpened():
        # read frame
        ret, image = cap.read()
        if not ret:
            break

        # Replace slow PIL conversion with direct numpy operations
        image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
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
        frame_count += 1
        global_frame_count += 1


end_time = time.time()
elapsed_time = end_time - start_time
processing_fps = total_frame_count / elapsed_time
print("Processed {} frames in {:.2f} seconds (Processing FPS: {:.2f})".format(total_frame_count, total_frame_count, processing_fps))

cap.release()
cv2.destroyAllWindows()
