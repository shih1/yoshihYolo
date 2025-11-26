# YOLO Video Object Detector

A simple, clean, and **highly extensible** implementation of YOLO object detection for video analysis using only OpenCV.

Detects 80 common objects (people, animals, vehicles, etc.) in video files and outputs annotated videos with bounding boxes and labels.

---

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download YOLO model
./setup_yolo.sh

# 3. Run detection
python main.py your_video.mp4
```

---

## 📐 Architecture Overview

### Core Philosophy

This framework is designed to be:

- **Simple**: Single-file execution, minimal dependencies
- **Modular**: Easy to extend with new features
- **AI-Agent-Friendly**: Clear structure for automated extension

### File Structure

```
yolo-video-detector/
├── main.py              # Core detection pipeline (extend here)
├── config.py            # All configuration parameters
├── setup_yolo.sh        # Model downloader (supports multiple YOLO versions)
├── requirements.txt     # Python dependencies
├── yolo/                # YOLO model files (gitignored)
└── output/              # Generated videos (gitignored)
```

### Key Functions to Understand

**In `main.py`:**

1. **`load_yolo_model()`** - Loads YOLO network, classes, colors

   - Returns: `net, classes, colors, output_layers`
   - Extend: Add custom model loading, ensemble models

2. **`process_video(video_path, net, classes, colors, output_layers)`** - Main detection loop

   - Frame-by-frame processing
   - Runs YOLO inference
   - Draws bounding boxes
   - Extend: Add tracking, custom visualizations, data export

3. **`main()`** - Entry point and CLI argument parsing
   - Extend: Add new CLI options, batch processing

---

## 🔧 Configuration

All parameters in `config.py`:

```python
# Video settings
INPUT_VIDEO = 'video.mp4'
MAX_FRAMES = None

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5    # Lower = more detections
NMS_THRESHOLD = 0.4           # Non-maximum suppression

# Performance
USE_GPU = False               # Enable CUDA acceleration
INPUT_SIZE = 416              # YOLO input size (320/416/608)

# Visualization
DRAW_BOXES = True
DRAW_LABELS = True
DRAW_CONFIDENCE = True
```

---

## 🚀 Extension Ideas for AI Agents

Here are concrete ways to extend this framework. Each includes implementation guidance.

### 1. **Object Tracking** (Add Temporal Consistency)

**What:** Track detected objects across frames with unique IDs

**Why:** Know which detection is the same object over time

**Implementation:**

- Add tracking library: `pip install filterpy` (Kalman filter) or use OpenCV's built-in trackers
- After NMS in `process_video()`, match current detections to previous frame
- Assign consistent IDs and draw trails

**Files to modify:**

- `main.py`: Add tracking logic after NMS step
- `config.py`: Add `TRACKING_ENABLED = True`

**Suggested approach:**

```python
# In process_video(), after NMS
if TRACKING_ENABLED:
    tracked_objects = tracker.update(boxes, class_ids)
    # Draw IDs and trails
```

---

### 2. **JSON/CSV Data Export** (Structured Detection Data)

**What:** Export detection data to structured format for analysis

**Why:** Enable downstream analytics, database storage, ML training

**Implementation:**

- Create list of detections per frame
- Export to JSON or CSV after processing

**Files to modify:**

- `main.py`: Add `export_detections()` function
- `config.py`: Add `EXPORT_FORMAT = 'json'` or `'csv'`

**Output format:**

```json
{
  "video": "video.mp4",
  "frames": [
    {
      "frame_number": 1,
      "detections": [
        {"class": "person", "confidence": 0.95, "bbox": [x, y, w, h]},
        {"class": "car", "confidence": 0.87, "bbox": [x, y, w, h]}
      ]
    }
  ]
}
```

---

### 3. **Multi-Video Batch Processing** (Process Multiple Videos)

**What:** Process entire directories of videos automatically

**Why:** Scale to large datasets, automate workflows

**Implementation:**

- Accept directory path as input
- Iterate through video files
- Process each with progress tracking

**Files to modify:**

- `main.py`: Add `--batch` CLI argument and batch processing logic

**Example usage:**

```bash
python main.py --batch videos/ --output-dir processed/
```

---

### 4. **Custom Object Filters** (Focus on Specific Classes)

**What:** Only detect/display specific object classes

**Why:** Reduce noise, focus on relevant objects (e.g., only people, only vehicles)

**Implementation:**

- Add class whitelist/blacklist
- Filter detections before drawing

**Files to modify:**

- `config.py`: Add `FILTER_CLASSES = ['person', 'car', 'bicycle']`
- `main.py`: Skip drawing for filtered classes

---

### 5. **Heat Maps** (Visualize Detection Density)

**What:** Show where objects are most frequently detected

**Why:** Spatial analysis, traffic patterns, crowd density

**Implementation:**

- Accumulate detection positions across frames
- Generate heat map overlay
- Blend with final frame or export separately

**Files to modify:**

- `main.py`: Add `generate_heatmap()` function
- Track detection centers in 2D histogram

**Output:** Static image showing detection hotspots

---

### 6. **Real-Time Webcam Detection** (Live Video Feed)

**What:** Run YOLO on webcam/RTSP stream in real-time

**Why:** Live monitoring, security cameras, interactive demos

**Implementation:**

- Replace `cv2.VideoCapture(file)` with `cv2.VideoCapture(0)` for webcam
- Add frame skipping for performance
- Display with `cv2.imshow()`

**Files to modify:**

- `main.py`: Add `--webcam` flag

**Example usage:**

```bash
python main.py --webcam
```

---

### 7. **Alert System** (Notifications on Detection)

**What:** Trigger alerts when specific objects are detected

**Why:** Security monitoring, anomaly detection

**Implementation:**

- Check if target class detected
- Send notification (email, SMS, webhook, sound)

**Files to modify:**

- `main.py`: Add alert logic after detection
- `config.py`: Add `ALERT_CLASSES = ['person']` and notification settings

**Example:**

```python
if 'person' in detected_classes and ALERT_ENABLED:
    send_email_alert(frame, timestamp)
```

---

### 8. **Zone-Based Detection** (Detect Only in ROI)

**What:** Only detect objects within defined regions of interest

**Why:** Ignore irrelevant areas, reduce false positives

**Implementation:**

- Define polygons/rectangles as zones
- Mask image before YOLO or filter detections by zone

**Files to modify:**

- `config.py`: Add `DETECTION_ZONES = [(x1,y1,x2,y2), ...]`
- `main.py`: Check if detection center is within zone

---

### 9. **Multi-Model Ensemble** (Combine Multiple YOLO Models)

**What:** Run multiple YOLO models and merge results

**Why:** Higher accuracy, reduce false negatives

**Implementation:**

- Load multiple models (e.g., YOLOv3 + YOLOv4)
- Run both on same frame
- Merge detections with weighted voting

**Files to modify:**

- `main.py`: Load multiple models in `load_yolo_model()`
- Combine detections before NMS

---

### 10. **Performance Profiling** (Measure Speed Bottlenecks)

**What:** Time each step (load, inference, NMS, drawing)

**Why:** Identify optimization opportunities

**Implementation:**

- Add timing decorators
- Print breakdown per frame

**Files to modify:**

- `main.py`: Add `@timing_decorator` or manual `time.time()` checks

**Output:**

```
Frame 100:
  Inference: 45ms
  NMS: 3ms
  Drawing: 2ms
  Total: 50ms (20 FPS)
```

---

### 11. **Custom Trained Models** (Detect Non-COCO Objects)

**What:** Use YOLO models trained on custom datasets

**Why:** Detect domain-specific objects (anime characters, medical imagery, etc.)

**Implementation:**

- Train custom YOLO model on your data
- Replace model files and class names

**Files to modify:**

- `config.py`: Point to custom `.weights`, `.cfg`, `.names`
- No code changes needed!

---

### 12. **Object Counting** (Count Objects Per Class)

**What:** Track unique object counts entering/exiting zones

**Why:** Traffic counting, crowd analytics

**Implementation:**

- Combine with tracking (#1)
- Define entry/exit lines
- Count crossings

**Files to modify:**

- `main.py`: Add line-crossing detection logic

---

### 13. **Thumbnail Generation** (Extract Key Frames)

**What:** Save frames with detections as thumbnails

**Why:** Quick preview, dataset creation

**Implementation:**

- Save frame to disk when detection occurs
- Optional: Only save high-confidence detections

**Files to modify:**

- `main.py`: Add `cv2.imwrite()` when detections found
- `config.py`: Add `SAVE_THUMBNAILS = True`

---

### 14. **Video Summarization** (Speed Up Based on Activity)

**What:** Create time-lapse that slows down during detections

**Why:** Watch hours of footage quickly while catching important events

**Implementation:**

- Skip frames when no detections
- Include all frames when objects present

**Files to modify:**

- `main.py`: Conditionally write frames to output

---

### 15. **API Server** (REST API for Detection Service)

**What:** Wrap detector in Flask/FastAPI web service

**Why:** Remote access, integrate with other systems

**Implementation:**

- Create `server.py` with Flask
- Accept video upload via POST
- Return JSON with detections

**New files:**

- `server.py`: Flask app
- Update `requirements.txt`: Add `flask`

**Example endpoint:**

```
POST /detect
Body: video file
Response: JSON with detections
```

---

## 🛠️ How to Implement Extensions

### Step-by-Step Guide for AI Agents

1. **Identify extension point:**

   - Read `main.py` to understand pipeline
   - Find where to inject new logic (usually in `process_video()`)

2. **Add configuration:**

   - Add new parameters to `config.py`
   - Use descriptive names with defaults

3. **Implement feature:**

   - Modify `main.py` functions or add new ones
   - Keep core pipeline intact
   - Add error handling

4. **Test:**

   - Run on sample video
   - Verify output
   - Check performance

5. **Document:**
   - Update this README
   - Add usage examples
   - Note any new dependencies

---

## 📊 Current Capabilities

**Detected Objects (80 COCO Classes):**

| Category     | Objects                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **People**   | person                                                                                                                             |
| **Animals**  | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe                                                                  |
| **Vehicles** | bicycle, car, motorcycle, airplane, bus, train, truck, boat                                                                        |
| **Indoor**   | chair, couch, bed, dining table, toilet, TV, laptop, keyboard, mouse, cell phone, microwave, oven, refrigerator, book, clock, vase |
| **Sports**   | frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket                                    |
| **Food**     | bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake              |

See `yolo/coco.names` for complete list.

---

## 🎯 YOLO Model Comparison

| Model           | Size    | Speed       | Accuracy           | Best For                                       |
| --------------- | ------- | ----------- | ------------------ | ---------------------------------------------- |
| **YOLOv4-tiny** | ~23 MB  | ⚡⚡⚡ Fast | ⭐⭐ Good          | Quick testing, real-time, resource-constrained |
| **YOLOv3**      | ~236 MB | ⚡⚡ Medium | ⭐⭐⭐ Great       | Balanced performance                           |
| **YOLOv4**      | ~246 MB | ⚡ Slower   | ⭐⭐⭐⭐ Excellent | High accuracy, offline processing              |

Download any model with `./setup_yolo.sh`

---

## 🐛 Common Issues & Solutions

### No objects detected

**Solutions:**

- Lower `CONFIDENCE_THRESHOLD` in `config.py` (try 0.2-0.3)
- Use larger model (YOLOv4 instead of tiny)
- Ensure video contains COCO classes
- Note: YOLO struggles with anime/cartoons (trained on real photos)

### Slow processing

**Solutions:**

- Use YOLOv4-tiny
- Reduce `INPUT_SIZE` to 320
- Set `MAX_FRAMES` to process fewer frames
- Enable GPU: `USE_GPU = True` (requires CUDA OpenCV)
- Skip frames: Only process every Nth frame

### Out of memory

**Solutions:**

- Process video in chunks
- Reduce `INPUT_SIZE`
- Close other applications
- Use YOLOv4-tiny

---

## 💻 CLI Usage Examples

```bash
# Basic usage
python main.py video.mp4

# Process only first 300 frames
python main.py video.mp4 --max-frames 300

# Lower confidence threshold
python main.py video.mp4 --confidence 0.3

# Use specific model
python main.py video.mp4 --model yolov4

# Combine options
python main.py video.mp4 --max-frames 500 --confidence 0.4 --model yolov3
```

---

## 🔬 Technical Details

### Detection Pipeline

```
1. Load video frame
2. Preprocess: Resize to 416x416, normalize to [0,1]
3. YOLO inference: Forward pass through network
4. Post-process: Extract bounding boxes, confidences, class IDs
5. NMS: Remove overlapping detections
6. Draw: Render boxes and labels
7. Write output frame
```

### Performance Tips

- **GPU acceleration**: 5-10x speedup with CUDA
- **Input size**: 320 = fast, 416 = balanced, 608 = accurate
- **Frame skipping**: Process every 2nd or 3rd frame for 2-3x speedup
- **Model choice**: Tiny models are 4-5x faster than full models

---

## 📦 Dependencies

**Required:**

- `opencv-python` - Computer vision operations
- `numpy` - Numerical operations

**Optional:**

- CUDA-enabled OpenCV for GPU acceleration
- `filterpy` for Kalman filtering (if implementing tracking)

---

## 🤝 Contributing

When extending this framework:

1. Keep the core simple
2. Add configuration to `config.py`
3. Document new features in this README
4. Test on multiple videos
5. Consider edge cases (empty frames, no detections, etc.)

---

## 📄 License

This project uses YOLO models which have their own licenses:

- YOLOv3/v4: [Darknet License](https://github.com/AlexeyAB/darknet)
- Models trained on COCO dataset

---

## 🎓 Learning Resources

**YOLO:**

- Original YOLO paper: "You Only Look Once: Unified, Real-Time Object Detection"
- Darknet: https://github.com/AlexeyAB/darknet
- YOLO explained: https://arxiv.org/abs/1506.02640

**OpenCV:**

- DNN module: https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html
- Video I/O: https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html

**COCO Dataset:**

- Classes and annotations: https://cocodataset.org/

---

## 🚀 Quick Extension Template

```python
# Example: Add custom feature to main.py

def my_custom_feature(frame, detections):
    """
    Your custom logic here

    Args:
        frame: Current video frame (numpy array)
        detections: List of (bbox, class, confidence) tuples

    Returns:
        modified_frame: Frame with your additions
    """
    # Your code here
    return frame

# In process_video(), after NMS:
if ENABLE_MY_FEATURE:  # Add to config.py
    frame = my_custom_feature(frame, current_detections)
```

---

**Last Updated:** 2024  
**Version:** 1.0 - Extensible Base Framework

_This README is designed to enable AI agents and developers to quickly understand, use, and extend this YOLO detection framework._
