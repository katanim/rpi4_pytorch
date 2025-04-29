import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import list_models, get_model
import cv2
from PIL import Image
import time
import ast


print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA devices found.")

# Load classes once, outside the loop
with open("mobilevnet_classes.txt", "r") as f:
    classes = ast.literal_eval(f.read())

# Open video file
video_path = "car-detection.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# List available models
all_models = list_models()
classification_models = list_models(module=torchvision.models)

# Initialize a model
model = get_model("mobilenet_v3_large", weights="IMAGENET1K_V1")
# model = models.quantization.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define image transformation
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        pil_image = Image.fromarray(image)
        input_tensor = preprocess(pil_image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        input_batch = input_batch.to(device)
        output = model(input_batch)

        # Process and display the top prediction
        probabilities = output[0].softmax(dim=0)
        idx = probabilities.argmax().item()
        confidence = probabilities[idx].item() * 100
        detected_object = classes[idx]

        if confidence > 25:  # Threshold for detection
            # print(f"Detected: {detected_object} with confidence {confidence:.2f}% at frame {global_frame_count}")
            print(f"Detected at frame {global_frame_count}")

        # log model performance
        frame_count += 1
        global_frame_count += 1

        # cv2.imshow("Frame", image)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

end_time = time.time()
elapsed_time = end_time - start_time
processing_fps = total_frame_count / elapsed_time
print("Processed {} frames in {:.2f} seconds (Processing FPS: {:.2f})".format(total_frame_count, total_frame_count, processing_fps))

cap.release()
cv2.destroyAllWindows()