# EH CVAT Mask to YOLO Segmentation Converter

This tool converts mask images to YOLO segmentation format. It processes both binary masks and masks with transparency, extracting contours and saving them in the normalized format required by YOLO.

## Prerequisites

- Python 3.11
- pip (Python package installer)

## Setup

1. Clone this repository or download the source code to your local machine.

2. Navigate to the project directory:
   ```
   cd path_to_code/eh_dataset_scripts
   ```

3. Create a virtual environment:
   ```
   python3 -m venv racoonvenv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your mask images are in PNG format and located in a single directory.

2. Run the script with the following command:
   ```
   python eh_mask_image_to_yolo_converter.py /path/to/input/directory /path/to/output/directory
   python eh_mask_image_to_yolo_converter.py ../eh_driver_window_dataset/labels/masks/ ../eh_driv
er_window_dataset/labels/train/
   ```
   Replace `/path/to/input/directory` with the path to your mask images, and `/path/to/output/directory` with the desired location for the YOLO format annotation files.

3. The script will process all PNG files in the input directory and save the corresponding YOLO format annotations in the output directory.

4. There is another script to verify if the output generated is correct. Run the following command to convert the YOLO format annotations back to mask images:
   ```sh
   python eh_yolo_to_mask_converter.py --txt_dir ../eh_seg_001_dataset/labels/train/ --img_dir ../eh_seg_001_dataset/images/train/ --output_dir ./temp_output_masks -v
   ```

## Output

The script generates text files in the YOLO segmentation format. Each line in the output file represents one object and follows this structure:

```
class_id x1 y1 x2 y2 ... xn yn
```

Where:
- `class_id` is always 0 in this implementation
- `x1 y1 x2 y2 ... xn yn` are the normalized coordinates of the polygon vertices
