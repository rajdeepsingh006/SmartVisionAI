import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
import tempfile
import os
from collections import defaultdict

WEAPON_CLASSES = ["knife", "gun", "pistol", "rifle"]
SUSPICIOUS_ITEMS = ["backpack", "bag"]

# Constants
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4#Non max suppression is a technique 
#used mainly in object detection that aims at selecting 
# the best bounding box out of a set of overlapping boxes.
LOITERING_TIME_THRESHOLD = 30
AGGRESSIVE_MOVEMENT_THRESHOLD = 0.3
ALERT_COOLDOWN = 30

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    weights_path = "yolov4-tiny.weights"
    config_path = "yolov4-tiny.cfg"
    
    if not os.path.exists(weights_path):
        st.error("YOLO weights file not found. Please download yolov4-tiny.weights.")
        st.stop()
    
    net = cv2.dnn.readNet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

# Detection and alert functions
def detect_fights(frame, prev_frame, motion_history, fgbg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if prev_frame is None:
        return False, 0, gray, motion_history
    
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fight_detected = False
    motion_intensity = 0
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
            
        (x, y, w, h) = cv2.boundingRect(contour)
        motion_intensity += w * h
        
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 2.0:
            motion_history.append((x, y, w, h))
            if len(motion_history) > 10:
                dx = np.std([m[0] for m in motion_history[-10:]])
                dy = np.std([m[1] for m in motion_history[-10:]])
                if dx > 15 and dy > 15:
                    fight_detected = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    motion_level = motion_intensity / (frame.shape[0] * frame.shape[1])
    return fight_detected, motion_level, gray, motion_history

def draw_detections(frame, detections, alerts):
    for detection in detections:
        label = detection['label']
        confidence = detection['confidence']
        box = detection['box']
        
        color = (0, 255, 0)
        if label in WEAPON_CLASSES:
            color = (0, 0, 255)
        elif label in SUSPICIOUS_ITEMS:
            color = (0, 165, 255)
        
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (box[0], box[1]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    for alert in alerts:
        if alert['type'] == 'weapon_detected':
            box = alert['box']
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 3)
            cv2.putText(frame, f"WEAPON: {alert['label'].upper()}", 
                       (box[0], box[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if alert['type'] == 'fight_detected':
            cv2.putText(frame, "FIGHT DETECTED!", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return frame

# Streamlit app
def main():
    st.title("Crime Prevention System")
    st.sidebar.title("Settings")
    
    input_source = st.sidebar.radio("Input Source", ["Webcam", "Upload Video"])
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.1, 1.0, 0.4, 0.1)
    
    net, classes, output_layers = load_yolo_model()
    
    if input_source == "Webcam":
        st.write("Using Webcam...")
        cap = cv2.VideoCapture(1)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_file.name)
        else:
            st.warning("Please upload a video file.")
            return
    
    stframe = st.empty()
    prev_frame = None
    motion_history = []
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")
                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))
                    
                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'class_id': class_ids[i],
                    'label': classes[class_ids[i]],
                    'confidence': confidences[i],
                    'box': boxes[i]
                })
        
        fight_detected, motion_level, prev_frame, motion_history = detect_fights(
            frame, prev_frame, motion_history, fgbg
        )
        
        alerts = []
        if fight_detected:
            alerts.append({"type": "fight_detected", "level": motion_level})
        
        frame = draw_detections(frame, detections, alerts)
        stframe.image(frame, channels="BGR")
        
        frame_count += 1
    
    cap.release()

if __name__ == "__main__":
    main()
