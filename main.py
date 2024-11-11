import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import cv2
import datetime
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealTimeDetector")
        self.root.geometry("800x800")  # Set fixed geometry
        self.root.resizable(False, False)  # Disable window resizing

        # Define configuration constants
        self.confidence_threshold = 0.5
        self.box_color = (0, 255, 0)
        self.device = "cpu"  # Change to "mps" for M1 Macs
        self.video_source = 0  # Default to front camera
        self.model = YOLO("yolov8n.pt")
        self.frame_skip = 2
        self.frame_count = 0
        self.cap = None
        self.running = False
        self.logs = []

        # Create a notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill=tk.BOTH)

        # Create frames for each tab
        self.video_frame = tk.Frame(self.notebook)
        self.info_frame = tk.Frame(self.notebook)

        # Add frames to notebook
        self.notebook.add(self.video_frame, text="Video Detection")
        self.notebook.add(self.info_frame, text="Info")

        # Setup Video Detection tab
        self.setup_video_detection_tab()

        # Setup Info tab
        self.setup_info_tab()

    def setup_video_detection_tab(self):
        # Create a frame for the video display and control buttons
        video_container = tk.Frame(self.video_frame)
        video_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a frame for the buttons
        button_frame = tk.Frame(video_container)
        button_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create canvas for video display
        self.canvas = tk.Canvas(video_container, width=640, height=480)
        self.canvas.pack(side=tk.RIGHT)

        # Create buttons for control
        button_width = 20
        self.start_button = ttk.Button(button_frame, text="Start Detection", command=self.start_detection, width=button_width)
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Video Capture", command=self.stop_capture, width=button_width)
        self.stop_button.pack(pady=5)

        self.save_button = ttk.Button(button_frame, text="Save Logs", command=self.save_logs, width=button_width)
        self.save_button.pack(pady=5)

        # Create a dropdown for camera selection
        self.camera_var = tk.StringVar(value="Front Camera")
        self.camera_menu = ttk.Combobox(button_frame, textvariable=self.camera_var, width=button_width)
        self.camera_menu['values'] = ("Front Camera", "Back Camera")
        self.camera_menu.bind("<<ComboboxSelected>>", self.change_camera)
        self.camera_menu.pack(pady=5)

        # Create a text box for logs
        self.log_box = tk.Text(self.video_frame, height=10, state='disabled')
        self.log_box.pack(fill=tk.BOTH, padx=10, pady=10)

    def setup_info_tab(self):
        info_text = (
            "Object Detection App\tVersion 1.0\n\n"
            "Developed using Python-OpenCv and YOLOv8\n"
            "This app uses a pre-trained YOLOv8 (YOLOv8n.pt) model & CPU to perform real-time object detection.\n\n"
            "Credits:\n"
            "Development:\tAmir Faramarzpour 2024-2025\n"
            "Libraries and Tools:\n"
            "- OpenCV\n"
            "- YOLOv8\n"
            "- Tkinter\n"
            "- PIL\n"
            "- ttkthemes\n\n"


            "About OpenCV:\n"
            "OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. "
            "It contains a comprehensive set of tools for image processing, video capture, and analysis, enabling developers to build "
            "complex computer vision applications easily. OpenCV supports various programming languages and platforms, making it a popular "
            "choice for both academic and industrial projects.\n\n"
            "About YOLO:\n"
            "YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. "
            "YOLOv8, the latest version, improves upon previous versions with enhanced performance, accuracy, and speed. "
            "It can detect multiple objects within an image or video frame simultaneously, making it ideal for applications "
            "ranging from security systems to autonomous driving. YOLO's architecture allows it to process images in real-time, "
            "which is essential for applications requiring immediate response.\n\n"
            "Available object detection initialization options are:\nyolov8n.pt - nano (3.2M parameters)\nyolov8s.pt - small (11.2M parameters)\nyolov8m.pt - medium (25.9M parameters)\nyolov8l.pt - large (43.7M parameters)\nyolov8x.pt - extra-large (68.2M parameters)\n"
        )

        info_text_widget = tk.Text(self.info_frame, wrap=tk.WORD)
        info_text_widget.insert(tk.END, info_text)
        info_text_widget.config(state='disabled')
        info_text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.info_frame, command=info_text_widget.yview)
        info_text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def change_camera(self, event):
        selected_camera = self.camera_var.get()
        if selected_camera == "Front Camera":
            self.video_source = 0
        elif selected_camera == "Back Camera":
            self.video_source = 1

    def start_detection(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(self.video_source)
            self.process_frame()

    def stop_capture(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.canvas.delete("all")

    def save_logs(self):
        with open("detection.txt", "w") as file:
            file.writelines(self.logs)
        self.log_box.config(state='normal')
        self.log_box.insert(tk.END, "Logs saved to detection.txt\n")
        self.log_box.config(state='disabled')

    def process_frame(self):
        if self.running:
            start = datetime.datetime.now()
            ret, frame = self.cap.read()

            if ret:
                self.frame_count += 1
                if self.frame_count % self.frame_skip == 0:
                    resized_frame = cv2.resize(frame, (640, 480))

                    detections = self.model(resized_frame, device=self.device)
                    result = self.model(resized_frame)[0]

                    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
                    classes = np.array(result.boxes.cls.cpu(), dtype="int")
                    confidence = np.array(result.boxes.conf.cpu(), dtype="float")

                    for cls, bbox, conf in zip(classes, bboxes, confidence):
                        (x, y, x2, y2) = bbox
                        object_name = self.model.names[cls]
                        if conf < self.confidence_threshold:
                            continue
                        box_color = self.get_box_color(conf)
                        cv2.rectangle(resized_frame, (x, y), (x2, y2), box_color, 2)
                        cv2.putText(resized_frame, f"{object_name}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, box_color, 2)

                        # Log detection info
                        log_entry = f"Detected {object_name} with confidence {conf:.2f} at [{x}, {y}, {x2}, {y2}]\n"
                        self.logs.append(log_entry)
                        self.update_log_box(log_entry)

                    end = datetime.datetime.now()
                    total = (end - start).total_seconds()

                    fps = f"FPS: {1 / total:.2f}"
                    cv2.putText(resized_frame, fps, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                    img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img_tk = ImageTk.PhotoImage(image=img)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    self.canvas.img_tk = img_tk

                self.root.after(10, self.process_frame)

    def get_box_color(self, conf):
        if conf > 0.6:
            return (37, 245, 75)
        elif conf > 0.3:
            return (66, 224, 245)
        else:
            return (78, 66, 245)

    def update_log_box(self, log_entry):
        self.log_box.config(state='normal')
        self.log_box.insert(tk.END, log_entry)
        self.log_box.config(state='disabled')
        self.log_box.yview(tk.END)

if __name__ == "__main__":
    root = ThemedTk(theme="equilux")
    app = ObjectDetectionApp(root)
    root.mainloop()
