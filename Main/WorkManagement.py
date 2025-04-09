import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from io import BytesIO
import threading
import queue
from typing import Optional, Tuple
from threading import Thread
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize all session state variables at the beginning
def initialize_session_state():
    # Alert settings
    if 'alert_status' not in st.session_state:
        st.session_state.alert_status = None
    if 'alert_error' not in st.session_state:
        st.session_state.alert_error = None
    if 'alert_threshold' not in st.session_state:
        st.session_state.alert_threshold = int(os.getenv('ALERT_THRESHOLD', 1))  # Default to 1 worker
    if 'alert_cooldown' not in st.session_state:
        st.session_state.alert_cooldown = int(os.getenv('ALERT_COOLDOWN', 10))  # 10 minutes
    if 'absence_threshold' not in st.session_state:
        st.session_state.absence_threshold = int(os.getenv('ABSENCE_THRESHOLD', 10))  # 10 minutes
    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = 0
    if 'last_worker_detection_time' not in st.session_state:
        st.session_state.last_worker_detection_time = time.time()
    if 'use_env_credentials' not in st.session_state:
        st.session_state.use_env_credentials = True
    if 'alert_queues' not in st.session_state:
        st.session_state.alert_queues = []
    if 'worker_presence_log' not in st.session_state:
        st.session_state.worker_presence_log = []
    if 'shift_start_time' not in st.session_state:
        st.session_state.shift_start_time = "08:00"
    if 'shift_end_time' not in st.session_state:
        st.session_state.shift_end_time = "17:00"
    if 'work_zones' not in st.session_state:
        st.session_state.work_zones = []
    
    # Telegram credentials
    if 'telegram_bot_token' not in st.session_state:
        st.session_state.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', "")
    if 'telegram_chat_id' not in st.session_state:
        st.session_state.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', "")

# Call the initialization function at the start
initialize_session_state()

# Page configuration
st.set_page_config(
    page_title="Workforce Monitoring Pro",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Preset configurations for different work environments
PRESETS = {
    "Construction Site": {
        "confidence": 0.7,
        "min_height": 15,
        "nms": 0.3,
        "help": "Best for outdoor construction environments with varying distances"
    },
    "Factory Floor": {
        "confidence": 0.8,
        "min_height": 20,
        "nms": 0.4,
        "help": "Optimized for indoor industrial settings with consistent lighting"
    },
    "Office Space": {
        "confidence": 0.6,
        "min_height": 10,
        "nms": 0.2,
        "help": "For office environments with smaller detection areas"
    },
    "Warehouse": {
        "confidence": 0.75,
        "min_height": 25,
        "nms": 0.35,
        "help": "Large open spaces with potentially distant workers"
    }
}

# Modified model loading with proper CUDA support check
@st.cache_resource
def load_model():
    try:
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()
    
    # Check for CUDA support properly
    gpu_status = "⚠️ Using CPU"
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            gpu_status = "⚡ GPU Acceleration Enabled"
            if net.getPreferableBackend() != cv2.dnn.DNN_BACKEND_CUDA:
                gpu_status = "⚠️ CUDA not available - Using CPU"
        else:
            gpu_status = "⚠️ No CUDA devices found - Using CPU"
    except Exception:
        gpu_status = "⚠️ CUDA not supported in this OpenCV build - Using CPU"
    
    try:
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Failed to load class names: {str(e)}")
        st.stop()
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers, gpu_status

net, classes, output_layers, gpu_status = load_model()

# Enhanced Telegram Alert Function with more details
def send_telegram_alert(bot_token=None, chat_id=None, result_queue=None, alert_type="presence", count=0, duration=0):
    if not all([bot_token, chat_id, result_queue]):
        print("Error: Missing arguments")
        return

    try:
        if alert_type == "absence":
            message = f"🚨 WORKFORCE ALERT!\nNo workers detected for {duration} minutes in the designated area!"
        elif alert_type == "presence":
            message = f"👷 WORKFORCE UPDATE\nDetected {count} workers in the area"
        else:
            message = "⚠️ Workforce monitoring alert"
            
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        params = {
            'chat_id': chat_id,
            'text': message
        }

        response = requests.post(url, params=params)
        if response.status_code == 200:
            result_queue.put(('success', "Telegram alert sent successfully"))
        else:
            result_queue.put(('error', f"Telegram API error: {response.text}"))
    except Exception as e:
        result_queue.put(('error', f"Exception occurred: {str(e)}"))

# Function to check if current time is within working hours
def is_within_working_hours():
    try:
        now = datetime.now().time()
        start_time = datetime.strptime(st.session_state.shift_start_time, "%H:%M").time()
        end_time = datetime.strptime(st.session_state.shift_end_time, "%H:%M").time()
        
        if start_time <= end_time:
            return start_time <= now <= end_time
        else:  # Overnight shift
            return now >= start_time or now <= end_time
    except:
        return True  # If there's an error in time parsing, assume always working hours

# Sidebar controls
with st.sidebar:
    st.title("Workforce Monitoring Settings")
    st.caption(gpu_status)
    
    # Quick settings presets
    preset = st.selectbox(
        "Environment Preset",
        options=list(PRESETS.keys()),
        index=0,
        help="Select a preset configuration for different work environments"
    )
    
    # Show preset description
    st.caption(PRESETS[preset]["help"])
    
    # Apply selected preset
    current_preset = PRESETS[preset]
    
    # Main controls
    input_method = st.radio(
        "Input Method",
        ["Upload File", "Use Camera"],
        index=0,
        help="Choose between uploading a file or using your camera"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=current_preset["confidence"], 
        step=0.05,
        help="Higher values reduce false positives but may miss some workers"
    )
    min_person_height = st.slider(
        "Minimum Worker Height (%)", 
        min_value=5, 
        max_value=30, 
        value=current_preset["min_height"], 
        step=1,
        help="Filters out small detections that are likely not workers"
    )
    nms_threshold = st.slider(
        "Overlap Threshold (NMS)", 
        min_value=0.1, 
        max_value=0.5, 
        value=current_preset["nms"], 
        step=0.05,
        help="Controls how much overlapping boxes are suppressed"
    )
    
    # Work schedule configuration
    st.markdown("---")
    st.subheader("Work Schedule")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.shift_start_time = st.text_input("Shift Start Time (HH:MM)", value=st.session_state.shift_start_time)
    with col2:
        st.session_state.shift_end_time = st.text_input("Shift End Time (HH:MM)", value=st.session_state.shift_end_time)
    
    # Alert configuration
    st.markdown("---")
    st.subheader("Alert Settings")
    
    enable_telegram = st.checkbox("Enable Telegram Alerts", value=False)
    
    if enable_telegram:
        st.session_state.use_env_credentials = st.radio(
            "Telegram Credentials Source",
            ["Use .env file", "Enter manually"],
            index=0 if st.session_state.use_env_credentials else 1
        )

        if st.session_state.use_env_credentials:
            st.info("Using credentials from .env file")
            st.write(f"Chat ID: {st.session_state.telegram_chat_id}")
        else:
            st.session_state.telegram_bot_token = st.text_input("Telegram Bot Token", type="password")
            st.session_state.telegram_chat_id = st.text_input("Telegram Chat ID")
    
    # Alert thresholds
    st.session_state.alert_threshold = st.slider(
        "Minimum workers required", 
        0, 50, st.session_state.alert_threshold,
        help="Alert if worker count falls below this number"
    )
    st.session_state.absence_threshold = st.slider(
        "Minutes without workers to alert", 
        1, 120, st.session_state.absence_threshold,
        help="Send alert if no workers detected for this duration"
    )
    st.session_state.alert_cooldown = st.slider(
        "Minutes between alerts", 
        1, 60, st.session_state.alert_cooldown,
        help="Prevent alert flooding with this cooldown period"
    )

# Main title and description
st.title("👷 Workforce Presence Monitoring")
st.markdown("""
    <style>
    .big-font {
        font-size:16px !important;
    }
    </style>
    <p class="big-font">Monitor worker presence in designated areas with real-time alerts when workers are absent or when minimum staffing levels aren't met.</p>
""", unsafe_allow_html=True)

# Enhanced people counting function with work zone detection
def count_workers(image: np.ndarray, confidence_thresh: float, 
                 min_height_percent: int, nms_thresh: float) -> Tuple[np.ndarray, int]:
    # Convert to 3 channels if needed
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    height, width = image.shape[:2]
    min_height = int(height * (min_height_percent / 100))
    
    try:
        blob = cv2.dnn.blobFromImage(
            image, 
            1/255.0, 
            (416, 416), 
            swapRB=True, 
            crop=False
        )
        net.setInput(blob)
        outs = net.forward(output_layers)
    except Exception as e:
        st.error(f"Error in detection: {str(e)}")
        return image, 0
    
    boxes = []
    confidences = []
    class_ids = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if class_id == 0 and confidence > confidence_thresh:  # Class 0 is 'person' in COCO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                if h >= min_height:
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    try:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)
    except Exception:
        indices = []
    
    result_image = image.copy()
    worker_count = 0
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            
            # Draw bounding box with different color based on confidence
            if confidences[i] > 0.8:
                color = (0, 255, 0)  # High confidence - green
            elif confidences[i] > 0.5:
                color = (0, 255, 255)  # Medium confidence - yellow
            else:
                color = (0, 0, 255)  # Low confidence - red
                
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                result_image, 
                f"Worker {confidences[i]:.1%}", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,
                color, 
                1
            )
            worker_count += 1
    
    # Improved count display with work zone information
    text = f"Workers: {worker_count}"
    font_scale = 4
    thickness = 15
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Create a semi-transparent background rectangle
    overlay = result_image.copy()
    cv2.rectangle(
        overlay, 
        (10, 10), 
        (20 + text_width, 15 + text_height), 
        (0, 0, 0), 
        -1
    )
    alpha = 0.6  # Transparency factor
    cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
    
    # Put the text
    cv2.putText(
        result_image, 
        text, 
        (15, 15 + text_height), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (0, 255, 0), 
        thickness
    )
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        result_image,
        timestamp,
        (width - 200, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return result_image, worker_count

# Helper function for download buttons
def get_download_button(data: bytes, filename: str, file_type: str) -> None:
    if file_type == "image":
        st.download_button(
            label="📥 Download Image",
            data=data,
            file_name=filename,
            mime="image/jpeg"
        )
    else:  # video
        st.download_button(
            label="📥 Download Video",
            data=data,
            file_name=filename,
            mime="video/mp4"
        )

# Camera capture function with enhanced worker monitoring
def camera_capture():
    st.subheader("Live Workforce Monitoring")
    with st.expander("Camera Settings", expanded=False):
        cam_resolution = st.selectbox("Resolution", ["640x480", "1280x720", "1920x1080"], index=1)
        width, height = map(int, cam_resolution.split("x"))
        rotate_camera = st.checkbox("Rotate Camera 180°", value=False)
    
    run_camera = st.checkbox("Start Monitoring", value=False, key="camera_toggle")
    FRAME_WINDOW = st.empty()
    
    try:
        camera = cv2.VideoCapture(1)  # Using default camera
        if not camera.isOpened():
            st.error("Failed to access camera")
            return
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Placeholder for stats
    stats_placeholder = st.empty()
    fps_counter = 0
    start_time = time.time()
    worker_counts = []
    result_queue = queue.Queue()
    absence_start_time = None
    
    while run_camera:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame")
            break
            
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]  # Drop alpha channel if present
    
            processed_frame, count = count_workers(
                frame,
                confidence_threshold,
                min_person_height,
                nms_threshold
                                )
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            break
        
        # Update worker presence tracking
        current_time = time.time()
        if count > 0:
            st.session_state.last_worker_detection_time = current_time
            absence_start_time = None
            # Log worker presence
            st.session_state.worker_presence_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "count": count,
                "status": "present"
            })
        else:
            if absence_start_time is None:
                absence_start_time = current_time
            # Log worker absence
            st.session_state.worker_presence_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "count": 0,
                "status": "absent"
            })
        
        # Check for absence alert
        absence_duration = 0
        if absence_start_time is not None:
            absence_duration = (current_time - absence_start_time) / 60  # in minutes
            
            if (absence_duration >= st.session_state.absence_threshold and 
                is_within_working_hours() and
                time.time() - st.session_state.last_alert_time > st.session_state.alert_cooldown * 60):
                
                bot_token = st.session_state.get("telegram_bot_token", None)
                chat_id = st.session_state.get("telegram_chat_id", None)

                if bot_token and chat_id:
                    alert_queue = queue.Queue()
                    threading.Thread(
                        target=send_telegram_alert,
                        args=(bot_token, chat_id, alert_queue, "absence", 0, int(absence_duration))
                    ).start()
                    
                    st.session_state.alert_queues.append(alert_queue)
                    st.session_state.last_alert_time = time.time()
                    st.toast(f"⚠️ No workers detected for {int(absence_duration)} minutes!")
                else:
                    st.warning("Telegram credentials missing")
        
        # Check for minimum worker threshold alert
        if (count < st.session_state.alert_threshold and 
            is_within_working_hours() and
            time.time() - st.session_state.last_alert_time > st.session_state.alert_cooldown * 60):
            
            bot_token = st.session_state.get("telegram_bot_token", None)
            chat_id = st.session_state.get("telegram_chat_id", None)

            if bot_token and chat_id:
                alert_queue = queue.Queue()
                threading.Thread(
                    target=send_telegram_alert,
                    args=(bot_token, chat_id, alert_queue, "presence", count, 0)
                ).start()
                
                st.session_state.alert_queues.append(alert_queue)
                st.session_state.last_alert_time = time.time()
                st.toast(f"⚠️ Only {count} workers detected (minimum is {st.session_state.alert_threshold})")
            else:
                st.warning("Telegram credentials missing")
        
        # Check for alert results
        try:
            status, message = result_queue.get_nowait()
            if status == 'success':
                st.session_state.alert_status = message
            else:
                st.session_state.alert_error = message
        except queue.Empty:
            pass
        
        # Calculate FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        fps = fps_counter / elapsed_time if elapsed_time > 0 else 0
        
        # Update stats
        worker_counts.append(count)
        avg_count = np.mean(worker_counts[-10:]) if worker_counts else 0
        
        # Display frame
        FRAME_WINDOW.image(processed_frame, channels="RGB")
        
        # Update stats display with more information
        stats_placeholder.markdown(f"""
            **Live Workforce Stats:**
            - Current Workers: {count}
            - Average (last 10 frames): {avg_count:.1f}
            - FPS: {fps:.1f}
            - Last Detection: {datetime.fromtimestamp(st.session_state.last_worker_detection_time).strftime('%H:%M:%S')}
            - Absence Duration: {absence_duration:.1f} minutes
            - Shift Status: {'Active' if is_within_working_hours() else 'Off Hours'}
        """)
        
        time.sleep(max(0.05, 1/30 - (time.time() - start_time - elapsed_time)))
    
    camera.release()
    if not run_camera:
        st.success("Monitoring session ended")

# File processing function with worker monitoring features
def file_processing():
    uploaded_file = st.file_uploader(
        "Upload work area image/video", 
        type=["jpg", "jpeg", "png", "mp4", "mov"],
        help="Select an image or video file to analyze for worker presence"
    )

    if not uploaded_file:
        return

    file_ext = uploaded_file.name.split(".")[-1].lower()
    result_queue = queue.Queue()
    
    if file_ext in ["jpg", "jpeg", "png"]:
        with st.spinner("Analyzing work area..."):
            try:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                tab1, tab2 = st.tabs(["Original", "Processed"])
                
                with tab1:
                    st.image(image, use_column_width=True, caption="Original Image")
                
                with tab2:
                    result_img, count = count_workers(
                        image_np, 
                        confidence_threshold, 
                        min_person_height, 
                        nms_threshold
                    )
                    result_pil = Image.fromarray(result_img)
                    st.image(result_pil, use_column_width=True, caption=f"Detected {count} workers")
                    
                    # Alert logic for images
                    if enable_telegram:
                        if count < st.session_state.alert_threshold:
                            Thread(target=send_telegram_alert, args=(
                                st.session_state.telegram_bot_token,
                                st.session_state.telegram_chat_id,
                                result_queue,
                                "presence",
                                count,
                                0
                            )).start()
                            st.toast(f"⚠️ Only {count} workers detected (minimum is {st.session_state.alert_threshold})")
                    
                    # Download button
                    buffered = BytesIO()
                    result_pil.save(buffered, format="JPEG", quality=90)
                    get_download_button(buffered.getvalue(), f"processed_{uploaded_file.name}", "image")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    elif file_ext in ["mp4", "mov"]:
        st.info("Video processing may take some time depending on length and settings")
        
        try:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tfile:
                tfile.write(uploaded_file.read())
                temp_input_path = tfile.name
            
            # Process video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
                temp_output_path = temp_output.name
            
            cap = cv2.VideoCapture(temp_input_path)
            if not cap.isOpened():
                st.error("Failed to open video file")
                return
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                temp_output_path, 
                fourcc, 
                fps, 
                (frame_width, frame_height)
            )
            
            # Progress bar and stats
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_count = 0
            worker_counts = []
            absence_start_time = None
            
            # Preview placeholder
            preview_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    processed_frame, count = count_workers(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        confidence_threshold,
                        min_person_height,
                        nms_threshold
                    )
                    out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                    
                    # Update worker presence tracking
                    current_time = time.time()
                    if count > 0:
                        st.session_state.last_worker_detection_time = current_time
                        absence_start_time = None
                    else:
                        if absence_start_time is None:
                            absence_start_time = current_time
                    
                    # Check for absence alert
                    absence_duration = 0
                    if absence_start_time is not None:
                        absence_duration = (current_time - absence_start_time) / 60  # in minutes
                        
                        if (absence_duration >= st.session_state.absence_threshold and 
                            is_within_working_hours() and
                            time.time() - st.session_state.last_alert_time > st.session_state.alert_cooldown * 60):
                            
                            bot_token = st.session_state.get("telegram_bot_token", None)
                            chat_id = st.session_state.get("telegram_chat_id", None)

                            if bot_token and chat_id:
                                alert_queue = queue.Queue()
                                threading.Thread(
                                    target=send_telegram_alert,
                                    args=(bot_token, chat_id, alert_queue, "absence", 0, int(absence_duration))
                                ).start()
                                
                                st.session_state.alert_queues.append(alert_queue)
                                st.session_state.last_alert_time = time.time()
                                st.toast(f"⚠️ No workers detected for {int(absence_duration)} minutes!")
                            else:
                                st.warning("Telegram credentials missing")
                    
                    # Check for minimum worker threshold alert
                    if (count < st.session_state.alert_threshold and 
                        is_within_working_hours() and
                        time.time() - st.session_state.last_alert_time > st.session_state.alert_cooldown * 60):
                        
                        bot_token = st.session_state.get("telegram_bot_token", None)
                        chat_id = st.session_state.get("telegram_chat_id", None)

                        if bot_token and chat_id:
                            alert_queue = queue.Queue()
                            threading.Thread(
                                target=send_telegram_alert,
                                args=(bot_token, chat_id, alert_queue, "presence", count, 0)
                            ).start()
                            
                            st.session_state.alert_queues.append(alert_queue)
                            st.session_state.last_alert_time = time.time()
                            st.toast(f"⚠️ Only {count} workers detected (minimum is {st.session_state.alert_threshold})")
                        else:
                            st.warning("Telegram credentials missing")
                    
                    # Update stats
                    frame_count += 1
                    worker_counts.append(count)
                    progress = int((frame_count / total_frames) * 100)
                    progress_bar.progress(min(progress, 100))
                    
                    # Show preview every 50 frames
                    if frame_count % 50 == 1:
                        preview_placeholder.image(processed_frame, caption=f"Frame {frame_count}/{total_frames}")
                    
                    # Update status
                    status_text.markdown(f"""
                        **Processing Status:**
                        - Frames Processed: {frame_count}/{total_frames}
                        - Current Workers: {count}
                        - Average Workers: {np.mean(worker_counts):.1f}
                        - Absence Duration: {absence_duration:.1f} minutes
                        - Estimated Time Remaining: {(total_frames-frame_count)/(fps if fps > 0 else 30):.1f}s
                    """)
                
                except Exception as e:
                    st.error(f"Error processing frame {frame_count}: {str(e)}")
                    break
            
            cap.release()
            out.release()
            
            # Show results
            st.success("Video analysis complete!")
            
            # Display side-by-side comparison
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Video")
                st.video(uploaded_file)
            
            with col2:
                st.subheader("Processed Video")
                try:
                    with open(temp_output_path, "rb") as f:
                        video_data = f.read()
                        st.video(video_data)
                        get_download_button(video_data, f"processed_{uploaded_file.name}", "video")
                except Exception as e:
                    st.error(f"Error displaying processed video: {str(e)}")
        
        except Exception as e:
            st.error(f"Video processing error: {str(e)}")
        finally:
            # Clean up temp files
            if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

# Check for Telegram alert results
for q in list(st.session_state.alert_queues):
    try:
        status, message = q.get_nowait()
        if status == 'success':
            st.session_state.alert_status = message
        else:
            st.session_state.alert_error = message
        st.session_state.alert_queues.remove(q)
    except queue.Empty:
        pass

# Display Telegram alert status in the main thread
if st.session_state.alert_status:
    st.success(st.session_state.alert_status)
    st.session_state.alert_status = None

if st.session_state.alert_error:
    st.error(st.session_state.alert_error)
    st.session_state.alert_error = None

# Worker presence log display
if st.session_state.worker_presence_log:
    with st.expander("📋 Worker Presence Log", expanded=False):
        st.write("Last 50 detections:")
        # Display last 50 entries in reverse chronological order
        for entry in reversed(st.session_state.worker_presence_log[-50:]):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(entry["timestamp"])
            with col2:
                st.text(f"Workers: {entry['count']}")
            with col3:
                status_color = "green" if entry["status"] == "present" else "red"
                st.markdown(f"<span style='color: {status_color}'>{entry['status'].upper()}</span>", unsafe_allow_html=True)

# Main app logic
if input_method == "Use Camera":
    camera_capture()
else:
    file_processing()

# App info and documentation
with st.expander("ℹ️ About this App & Usage Guide", expanded=False):
    st.markdown("""
        ## Workforce Monitoring Pro
        
        This application uses advanced computer vision to monitor worker presence in designated areas, 
        providing real-time alerts when staffing levels fall below requirements or when workers are absent.

        ### Key Features:
        - **Worker Presence Detection**: Accurately counts workers in images, videos, or live camera feeds
        - **Absence Alerts**: Get notified when no workers are detected for specified durations
        - **Staffing Level Monitoring**: Alerts when worker count falls below minimum requirements
        - **Shift Scheduling**: Configure working hours to only monitor during active shifts
        - **Detailed Logging**: Maintains a record of all worker detections
        - **Multiple Environment Presets**: Optimized settings for different work environments
        - **Telegram Integration**: Receive alerts directly to your phone or team chat

        ### Usage Guide:
        1. **Select Environment Preset**: Choose the work environment that best matches your scenario
        2. **Configure Monitoring**:
           - Set minimum worker requirements
           - Configure absence alert thresholds
           - Set up working hours
        3. **Start Monitoring**:
           - Use live camera feed for real-time monitoring
           - Upload images/videos for analysis of recorded footage
        4. **Review Alerts & Logs**:
           - Check the sidebar for real-time stats
           - Review the worker presence log for historical data

        ### Technical Notes:
        - Requires YOLOv3 weights and config files
        - GPU acceleration automatically enabled if available
        - All processing happens locally - no data is uploaded
        - Telegram alerts require bot token and chat ID configuration
    """)

# Add footer
st.markdown("---")
st.caption("""
    Workforce Monitoring Pro | 
    [GitHub Repository](https://github.com/example/workforce-monitoring) | 
    Created with Streamlit
""")
