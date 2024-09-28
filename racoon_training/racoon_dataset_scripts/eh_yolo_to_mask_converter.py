import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

def yolo_boundary_to_mask(line, shape):
    """
    Convert a YOLO boundary format line to a binary mask.
    
    Parameters:
    - line: string representing the YOLO boundary format.
    - shape: tuple representing the shape of the mask (height, width).
    
    Returns:
    - Binary mask with the boundary filled in and the class ID.
    """
    # Parse the line (considering space-separated values)
    data = list(map(float, line.split()))
    class_id = int(data[0])
    coordinates = np.array(data[1:]).reshape(-1, 2)
    
    # Denormalize the coordinates
    height, width = shape
    denormalized_coordinates = (coordinates * [width, height]).astype(np.int32)
    
    # Create a blank mask
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Draw the contour and fill it
    cv2.drawContours(mask, [denormalized_coordinates], 0, (1), thickness=cv2.FILLED)
    
    return mask, class_id

def regenerate_mask_from_yolo(txt_path, shape, color_mapping):
    """
    Regenerate the segmentation mask from the YOLO boundary format text file.
    
    Parameters:
    - txt_path: path to the YOLO boundary format text file.
    - shape: tuple representing the shape of the mask (height, width).
    - color_mapping: dictionary mapping class IDs to RGB colors.
    
    Returns:
    - Segmentation mask with all classes.
    """
    # Create a blank RGB canvas
    mask_img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    
    # Read the text file and overlay each mask
    with open(txt_path, 'r') as file:
        for line in file:
            mask, class_id = yolo_boundary_to_mask(line, shape)
            mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            color = color_mapping.get(class_id, (255, 255, 255))  # Default to white if class not in mapping
            mask_colored = mask_rgb * np.array(color)
            mask_img = cv2.add(mask_img, mask_colored.astype(np.uint8))
            
    return mask_img

def yolo_txt_to_mask_images(txt_dir, img_dir, output_dir, color_mapping):
    """
    Convert YOLO format txt files to mask images.
    
    Parameters:
    - txt_dir: Directory containing the YOLO formatted txt files.
    - img_dir: Directory containing the original RGB images.
    - output_dir: Directory to save the regenerated mask images.
    - color_mapping: Dictionary mapping class IDs to RGB colors.
    """
    txt_dir = Path(txt_dir)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(txt_dir.glob("*.txt"))
    for txt_file in tqdm(txt_files, desc="Generating Masks"):
        img_file = img_dir / (txt_file.stem + ".jpg")
        
        if not img_file.exists():
            tqdm.write(f"Image not found for {img_file}. Skipping...")
            continue

        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Regenerate the mask
        regenerated_mask = regenerate_mask_from_yolo(str(txt_file), img.shape[:2], color_mapping)

        # Save the regenerated mask with the same name
        output_mask_path = output_dir / (txt_file.stem + ".png")
        cv2.imwrite(str(output_mask_path), cv2.cvtColor(regenerated_mask, cv2.COLOR_RGB2BGR))

def verify_yolo_segmentation(txt_dir, img_dir, output_dir, color_mapping):
    """
    Verify YOLO segmentation by overlaying masks on original images.
    
    Parameters:
    - txt_dir: Directory containing the YOLO formatted txt files.
    - img_dir: Directory containing the original RGB images.
    - output_dir: Directory to save the verification images.
    - color_mapping: Dictionary mapping class IDs to RGB colors.
    """
    txt_dir = Path(txt_dir)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(txt_dir.glob("*.txt"))
    for txt_file in tqdm(txt_files, desc="Verifying Segmentation"):
        img_file = img_dir / (txt_file.stem + ".jpg")
        
        if not img_file.exists():
            tqdm.write(f"Image not found for {img_file}. Skipping...")
            continue

        # Read the original image
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Regenerate the mask
        regenerated_mask = regenerate_mask_from_yolo(str(txt_file), img.shape[:2], color_mapping)

        # Overlay the mask on the original image
        alpha = 0.5  # Transparency factor
        overlay = cv2.addWeighted(img, 1 - alpha, regenerated_mask, alpha, 0)

        # Save the verification image
        output_path = output_dir / (txt_file.stem + "_verified.png")
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        tqdm.write(f"Verification image saved: {output_path}")

# Extended color mapping to include more classes
color_mapping = {
    0: (255, 0, 0),    # Red
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
    3: (255, 255, 0),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Cyan
    6: (128, 0, 0),    # Maroon
    7: (0, 128, 0),    # Green (dark)
    8: (0, 0, 128),    # Navy
    9: (128, 128, 0),  # Olive
    # Add more colors as needed for additional classes
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO format txt files to mask images and verify segmentation.")
    parser.add_argument("-t", "--txt_dir", required=True, help="Directory containing the YOLO formatted txt files.")
    parser.add_argument("-i", "--img_dir", required=True, help="Directory containing the original RGB images.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the regenerated mask images.")
    parser.add_argument("-v", "--verify", action="store_true", help="Verify segmentation by overlaying masks on original images.")
    args = parser.parse_args()
    
    if args.verify:
        verify_yolo_segmentation(args.txt_dir, args.img_dir, args.output_dir, color_mapping)
    else:
        yolo_txt_to_mask_images(args.txt_dir, args.img_dir, args.output_dir, color_mapping)