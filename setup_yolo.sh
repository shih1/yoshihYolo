#!/bin/bash

# YOLO Model Setup Script
# Downloads various YOLO models for object detection

set -e  # Exit on error

echo "========================================"
echo "  YOLO Model Downloader"
echo "========================================"
echo ""
echo "This script will download YOLO model files."
echo "Choose the model that best fits your needs."
echo ""

# Create yolo directory
mkdir -p yolo
cd yolo

# Function to download with progress
download_file() {
    local url=$1
    local filename=$2
    
    if [ -f "$filename" ]; then
        echo "✓ $filename already exists"
        return 0
    fi
    
    echo "Downloading $filename..."
    wget --progress=bar:force:noscroll "$url" -O "$filename"
    echo "✓ Downloaded $filename"
}

# Show menu
echo "Available YOLO models:"
echo ""
echo "1) YOLOv4-tiny    [~23 MB]  ⚡⚡⚡ Fast      ⭐⭐ Good"
echo "2) YOLOv3         [~236 MB] ⚡⚡ Medium     ⭐⭐⭐ Great"
echo "3) YOLOv4         [~246 MB] ⚡ Slower      ⭐⭐⭐⭐ Excellent"
echo "4) YOLOv7-tiny    [~12 MB]  ⚡⚡⚡ Fastest   ⭐⭐ Good"
echo "5) YOLOv7         [~75 MB]  ⚡⚡ Medium     ⭐⭐⭐⭐ Excellent (Recommended)"
echo "6) Download all models"
echo ""
read -p "Select model (1-6): " choice

case $choice in
    1)
        MODEL_NAME="YOLOv4-tiny"
        download_file "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights" "yolov4-tiny.weights"
        download_file "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg" "yolov4-tiny.cfg"
        WEIGHTS_FILE="yolov4-tiny.weights"
        CONFIG_FILE="yolov4-tiny.cfg"
        ;;
    2)
        MODEL_NAME="YOLOv3"
        download_file "https://pjreddie.com/media/files/yolov3.weights" "yolov3.weights"
        download_file "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg" "yolov3.cfg"
        WEIGHTS_FILE="yolov3.weights"
        CONFIG_FILE="yolov3.cfg"
        ;;
    3)
        MODEL_NAME="YOLOv4"
        download_file "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" "yolov4.weights"
        download_file "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" "yolov4.cfg"
        WEIGHTS_FILE="yolov4.weights"
        CONFIG_FILE="yolov4.cfg"
        ;;
    4)
        MODEL_NAME="YOLOv7-tiny"
        echo "Note: YOLOv7 uses PyTorch format. You'll need to convert or use a different implementation."
        echo "For now, downloading YOLOv4-tiny as fallback..."
        download_file "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights" "yolov4-tiny.weights"
        download_file "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg" "yolov4-tiny.cfg"
        WEIGHTS_FILE="yolov4-tiny.weights"
        CONFIG_FILE="yolov4-tiny.cfg"
        ;;
    5)
        MODEL_NAME="YOLOv7"
        echo "Note: YOLOv7 uses PyTorch format. You'll need to convert or use a different implementation."
        echo "For now, downloading YOLOv4 as best alternative..."
        download_file "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" "yolov4.weights"
        download_file "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" "yolov4.cfg"
        WEIGHTS_FILE="yolov4.weights"
        CONFIG_FILE="yolov4.cfg"
        ;;
    6)
        MODEL_NAME="All models"
        download_file "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights" "yolov4-tiny.weights"
        download_file "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg" "yolov4-tiny.cfg"
        download_file "https://pjreddie.com/media/files/yolov3.weights" "yolov3.weights"
        download_file "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg" "yolov3.cfg"
        download_file "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" "yolov4.weights"
        download_file "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" "yolov4.cfg"
        WEIGHTS_FILE="yolov4.weights"  # Default to v4
        CONFIG_FILE="yolov4.cfg"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Download class names if not present
if [ ! -f "coco.names" ]; then
    echo ""
    echo "Downloading COCO class names..."
    download_file "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" "coco.names"
fi

cd ..

# Update config.py with selected model
echo ""
echo "Updating config.py with selected model..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|YOLO_WEIGHTS = .*|YOLO_WEIGHTS = 'yolo/$WEIGHTS_FILE'|g" config.py
    sed -i '' "s|YOLO_CONFIG = .*|YOLO_CONFIG = 'yolo/$CONFIG_FILE'|g" config.py
else
    # Linux
    sed -i "s|YOLO_WEIGHTS = .*|YOLO_WEIGHTS = 'yolo/$WEIGHTS_FILE'|g" config.py
    sed -i "s|YOLO_CONFIG = .*|YOLO_CONFIG = 'yolo/$CONFIG_FILE'|g" config.py
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Model: $MODEL_NAME"
echo "Files downloaded to yolo/ directory"
echo ""
echo "You can now run:"
echo "  python main.py your_video.mp4"
echo ""
echo "To switch models later, run this script again"
echo "or manually edit config.py"
echo ""