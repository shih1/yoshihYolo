#!/usr/bin/env python3
"""
YOLO Video Object Detector
Simple object detection on video files using YOLO and OpenCV
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

from config import *

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("  YOLO VIDEO OBJECT DETECTOR")
    print("  Detect 80 common objects in video files")
    print("=" * 70)
    print()

def load_yolo_model():
    """Load YOLO model from config"""
    if not os.path.exists(YOLO_WEIGHTS):
        print(f"ERROR: YOLO weights not found at: {YOLO_WEIGHTS}")
        print("\nPlease run: ./setup_yolo.sh")
        print("to download the required model files.")
        sys.exit(1)
    
    if not os.path.exists(YOLO_CONFIG):
        print(f"ERROR: YOLO config not found at: {YOLO_CONFIG}")
        print("\nPlease run: ./setup_yolo.sh")
        sys.exit(1)
    
    if not os.path.exists(YOLO_NAMES):
        print(f"ERROR: Class names not found at: {YOLO_NAMES}")
        print("\nPlease run: ./setup_yolo.sh")
        sys.exit(1)
    
    print(f"Loading YOLO model...")
    print(f"  Weights: {YOLO_WEIGHTS}")
    print(f"  Config:  {YOLO_CONFIG}")
    
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
    
    # Set backend
    if USE_GPU:
        print("  Backend: CUDA (GPU)")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("  Backend: OpenCV (CPU)")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Load class names
    with open(YOLO_NAMES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Generate random colors for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    print(f"  Classes: {len(classes)} object types")
    print("✓ Model loaded successfully")
    print()
    
    return net, classes, colors, output_layers

def process_video(video_path, net, classes, colors, output_layers):
    """Process video and detect objects"""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        sys.exit(1)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fps = OUTPUT_FPS if OUTPUT_FPS else fps
    
    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    if MAX_FRAMES:
        print(f"  Processing: {min(MAX_FRAMES, total_frames)} frames")
    print()
    
    # Setup output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_name = Path(video_path).stem
    output_path = os.path.join(OUTPUT_DIR, f'{video_name}_detected.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # Statistics
    detection_counts = {cls: 0 for cls in classes}
    total_detections = 0
    frames_with_detections = 0
    frame_count = 0
    
    print("Processing video...")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"NMS threshold: {NMS_THRESHOLD}")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Check frame limit
        if MAX_FRAMES and frame_count > MAX_FRAMES:
            break
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        
        # Run detection
        detections = net.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    # Object detected
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                   CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        frame_detections = 0
        
        if len(indices) > 0:
            frames_with_detections += 1
            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                label = classes[class_ids[idx]]
                confidence = confidences[idx]
                color = colors[class_ids[idx]]
                
                # Draw bounding box
                if DRAW_BOXES:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
                
                # Draw label
                if DRAW_LABELS:
                    text = f"{label}"
                    if DRAW_CONFIDENCE:
                        text += f": {confidence:.2f}"
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
                    )
                    cv2.rectangle(frame, (x, y - text_height - 10),
                                (x + text_width, y), color, -1)
                    
                    cv2.putText(frame, text, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                               (0, 0, 0), FONT_THICKNESS)
                
                # Update statistics
                detection_counts[label] += 1
                total_detections += 1
                frame_detections += 1
        
        # Add frame info
        if SHOW_FRAME_INFO:
            info_text = f"Frame: {frame_count}"
            if MAX_FRAMES:
                info_text += f"/{MAX_FRAMES}"
            else:
                info_text += f"/{total_frames}"
            info_text += f" | Detections: {frame_detections}"
            
            cv2.putText(frame, info_text, FRAME_INFO_POSITION,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_count % PRINT_PROGRESS_EVERY == 0:
            progress = frame_count / (MAX_FRAMES if MAX_FRAMES else total_frames) * 100
            print(f"Processed {frame_count} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    # Print statistics
    if PRINT_STATISTICS:
        print()
        print("=" * 70)
        print("DETECTION STATISTICS")
        print("=" * 70)
        print(f"\nTotal detections: {total_detections}")
        print(f"Frames with detections: {frames_with_detections}/{frame_count}")
        print(f"Detection rate: {frames_with_detections/frame_count*100:.1f}%")
        
        print(f"\nDetections by class:")
        detected_classes = {k: v for k, v in detection_counts.items() if v > 0}
        if detected_classes:
            for cls, count in sorted(detected_classes.items(),
                                    key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {count}")
        else:
            print("  No objects detected")
            print("\n  Tip: Try lowering CONFIDENCE_THRESHOLD in config.py")
        
        print(f"\nOutput saved to: {output_path}")
        print("=" * 70)
    
    return output_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='YOLO Video Object Detector'
    )
    parser.add_argument('video', nargs='?', default=INPUT_VIDEO,
                       help='Input video file (default: from config.py)')
    parser.add_argument('--max-frames', type=int, default=MAX_FRAMES,
                       help='Maximum frames to process')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--model', choices=['yolov3', 'yolov4', 'yolov4-tiny'],
                       help='YOLO model to use (overrides config.py)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Override config with CLI args
    if args.max_frames:
        globals()['MAX_FRAMES'] = args.max_frames
    if args.confidence:
        globals()['CONFIDENCE_THRESHOLD'] = args.confidence
    if args.model:
        globals()['YOLO_WEIGHTS'] = f'yolo/{args.model}.weights'
        globals()['YOLO_CONFIG'] = f'yolo/{args.model}.cfg'
    
    # Check video exists
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)
    
    # Load model
    net, classes, colors, output_layers = load_yolo_model()
    
    # Process video
    output_path = process_video(args.video, net, classes, colors, output_layers)
    
    print("\nDone!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)