import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, time_synchronized, TracedModel
from utils.plots import plot_one_box

def perform_object_detection(image, model, device, img_size, conf_thres=0.25, iou_thres=0.45):
    # Function for object detection
    imgsz = check_img_size(img_size, s=model.stride.max())  # Check image size
    img = cv2.resize(image, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Convert image to torch tensor
    img = torch.from_numpy(img).to(device)
    img = img.half() if device.type != 'cpu' else img.float()  # Use FP16 if CUDA is available

    # Perform inference
    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # Apply non-maximum suppression
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # Draw bounding boxes and labels
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=(0,255,0), line_thickness=2)

    return image

def main():
    st.title("Object Detection with YOLOv7")

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the YOLO model
        weights = 'yolov7.pt'  # Adjust this path to your YOLOv7 weights file
        device = select_device('')  # Automatically select device (GPU if available, else CPU)
        model = attempt_load(weights, map_location=device)
        img_size = 640
        names = model.module.names if hasattr(model, 'module') else model.names

        # Read the uploaded image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Perform object detection
        output_image = perform_object_detection(image_np, model, device, img_size)

        # Display the result
        st.image(output_image, caption='Detected Objects', use_column_width=True)

if __name__ == "__main__":
    main()
