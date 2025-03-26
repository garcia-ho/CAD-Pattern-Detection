# CAD Matching Toolkit

This repository provides a comprehensive toolkit for CAD pattern detection and text matching in engineering documents.


## Features

### YOLO-Based Pattern Detection
- Detects engineering components in CAD drawings and P&IDs using YOLOv11
- Optimized for detecting multiple component types simultaneously
- Colorful annotations with unique colors for each component type
- High-resolution PDF processing with automatic tiling for large documents
- Supports concurrent processing on multi-core systems
- Generates detailed CSV reports of detection counts


### Text Recognition
- OCR processing for text extraction from PDF documents
- String matching capability for finding specific text in engineering documents
- Highlights text occurrences with distinct colors for easy visualization
- Processes high-resolution images in overlapping patches for optimal text detection
- Generates CSV reports of text occurrence counts



## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/CAD_matching.git
cd CAD_matching
pip install -r requirements.txt
```



## 1. YOLO-Based Pattern Detection

### Processing Pipeline
The pattern detection process works in two stages:

1. Preprocessing:

- Converts PDFs to high-resolution PNG images
- Applies binary filtering to optimize drawings for detection
- Handles large images via tiling to maintain full resolution

2. YOLO Detection:

- Segments large images for optimal detection
- Applies trained YOLO model to detect components
- Generates colorful annotations with component labels
- Creates CSV report with component counts



### Usage

1. Pattern Matching

```bash
cd CAD_matching
python Pattern_match/main.py --input-cad Pattern_match/input_CAD --processed-dir Pattern_match/input_image --output Pattern_match/output_CAD
```

<img src="readme_png/Valve_det.png" width="500" alt="Pattern detection example">

<img src="readme_png/example2.png" width="500" alt="Pattern detection example">

<img src="readme_png/example3.png" width="500" alt="Pattern detection example">


### Command-Line Options


| Option | Description | Default |
|--------|-------------|---------|
| `--input-cad` | Directory containing CAD drawings (PDF/images) | `Pattern_match/input_CAD` |
| `--processed-dir` | Directory for preprocessed images | `Pattern_match/input_image` |
| `--output` | Output directory for annotated images | `Pattern_match/output_CAD` |
| `--model` | Path to YOLO model weights | `runs/detect/train/weights/best.pt` |
| `--dpi` | Resolution for PDF conversion (dots per inch) | `400` |
| `--workers` | Number of worker threads for parallel processing | `4` |
| `--size` | YOLO detection segment size | `1024` |
| `--overlap` | Overlap between segments (0-1) | `0.15` |
| `--conf` | Confidence threshold (0-1) | `0.25` |
| `--iou` | IoU threshold (0-1) | `0.45` |
| `--report` | Path for CSV report | `Pattern_match/detection_report.csv` |
| `--skip-preprocess` | Skip preprocessing step | `False` |



-------------------

## 2. Text Recognition and Matching
The text recognition process consists of two main stages:

### 1. Preprocessing:

- Converts PDFs to high-resolution images
- Enhances text visibility with advanced filtering techniques
- Removes noise and optimizes contrast for better OCR results
- Text Detection and Matching:

### 2. Splits large images into overlapping patches for detailed text detection
### 3. Uses PaddleOCR to detect text in each patch
- Matches detected text against target words from a specified list
- Highlights and annotates matched words in the output images
- Generates detailed CSV reports with occurrence counts

```bash
cd CAD_matching
python String_match/main_text_det.py
```



<img src="readme_png/text_rec.png" width="500" alt="Pattern detection example">



## Output

- Visual output with highlighted patterns in output directory
- CSV report (valve_reports.csv) showing pattern counts for each file
- Text report summarizing detection statistics


