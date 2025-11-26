"""
YOLO Video Object Detector - Configuration
"""

# ============================================================================
# VIDEO SETTINGS
# ============================================================================

INPUT_VIDEO = 'video.mp4'    # Default input video (can be overridden via CLI)
MAX_FRAMES = None             # Maximum frames to process (None = all frames)
OUTPUT_DIR = 'output'         # Directory for output videos

# ============================================================================
# YOLO MODEL SETTINGS
# ============================================================================

# Model files (edit these after running setup_yolo.sh)
YOLO_WEIGHTS = 'yolo/yolov4-tiny.weights'
YOLO_CONFIG = 'yolo/yolov4-tiny.cfg'
YOLO_NAMES = 'yolo/coco.names'

# ============================================================================
# DETECTION SETTINGS
# ============================================================================

CONFIDENCE_THRESHOLD = 0.2    # Minimum confidence for detection (0.0 - 1.0)
                              # Lower = more detections but less accurate
                              # Higher = fewer but more confident detections
                              # Recommended: 0.3-0.6

NMS_THRESHOLD = 0.4           # Non-maximum suppression threshold (0.0 - 1.0)
                              # Lower = fewer overlapping boxes
                              # Higher = more overlapping boxes allowed
                              # Recommended: 0.3-0.5

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

USE_GPU = False               # Enable CUDA GPU acceleration
                              # Requires CUDA-enabled OpenCV build
                              # Can provide 5-10x speedup

INPUT_SIZE = 416              # YOLO input size (must be multiple of 32)
                              # Options: 320 (fast), 416 (balanced), 608 (accurate)
                              # Larger = slower but more accurate

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

DRAW_BOXES = True             # Draw bounding boxes
DRAW_LABELS = True            # Draw class labels
DRAW_CONFIDENCE = True        # Show confidence scores
BOX_THICKNESS = 2             # Bounding box line thickness
FONT_SCALE = 0.5              # Label font size
FONT_THICKNESS = 2            # Label font thickness

SHOW_FRAME_INFO = True        # Show frame number and detection count
FRAME_INFO_POSITION = (10, 30)  # Position of frame info text

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

OUTPUT_CODEC = 'mp4v'         # Video codec ('mp4v', 'XVID', 'H264')
OUTPUT_FPS = None             # Output FPS (None = same as input)

PRINT_PROGRESS_EVERY = 30     # Print progress every N frames
PRINT_STATISTICS = True       # Print detection statistics at end