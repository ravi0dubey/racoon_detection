import cv2
import numpy as np
import os
import glob
from pathlib import Path
import argparse
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiClassMaskToYOLOConverter:
    """
    MultiClassMaskToYOLOConverter is a utility class for converting multi-class mask images into YOLO segmentation format. 
    This class reads mask images where different colors represent different classes, extracts contours for each class, 
    and saves these contours in a normalized format suitable for YOLO segmentation.

    The conversion process involves:
    - Reading the mask image.
    - Identifying unique colors (classes) in the mask.
    - Extracting contours for each class.
    - Normalizing the contour coordinates relative to the image size.
    - Saving the normalized coordinates in a text file, following the YOLO segmentation format.

    Attributes:
        input_dir (str): Directory containing the original mask images.
        output_dir (str): Directory where the YOLO format annotation files will be saved.
        class_colors (dict): Dictionary mapping colors to class IDs.

    Methods:
        mask_to_yolo_segmentation(mask_path): Converts a single mask image to YOLO format.
        convert_directory_masks(): Processes all mask images in the input directory.
    """

    def __init__(self, input_dir, output_dir):
        """
        Initializes the MultiClassMaskToYOLOConverter with the input and output directory paths.

        :param input_dir: Directory containing mask images.
        :param output_dir: Directory to save the YOLO format annotation files.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.class_colors = {}  # Will be populated as we process images

    def get_class_id(self, color):
        """
        Get the class ID for a given color. If the color is new, assign a new class ID.

        :param color: RGB color tuple
        :return: Class ID
        """
        color_key = tuple(color)
        if color_key not in self.class_colors:
            self.class_colors[color_key] = len(self.class_colors)
        return self.class_colors[color_key]

    def mask_to_yolo_segmentation(self, mask_path):
        """
        Converts a single multi-class mask image to YOLO segmentation format.

        :param mask_path: Path to the mask image file.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.shape[2] == 4:  # If the image has an alpha channel
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)
        
        height, width = mask.shape[:2]
        unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)

        output_path = Path(self.output_dir) / (Path(mask_path).stem + '.txt')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as file:
            for color in unique_colors:
                if np.all(color == [0, 0, 0]):  # Skip black background
                    continue
                
                class_id = self.get_class_id(color)
                
                # Create a binary mask for this color
                color_mask = np.all(mask == color, axis=2).astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    normalized_coordinates = np.squeeze(approx) / [width, height]
                    flattened = normalized_coordinates.flatten().tolist()
                    file.write(f'{class_id} ' + ' '.join(map(str, flattened)) + '\n')

        logging.info(f"Processed {mask_path} to YOLO format.")

    def convert_directory_masks(self):
        """
        Converts all mask images in the input directory to YOLO segmentation format.
        """
        mask_files = glob.glob(os.path.join(self.input_dir, '*.png'))
        for mask_path in mask_files:
            self.mask_to_yolo_segmentation(mask_path)
        logging.info("Completed processing all mask images.")
        logging.info(f"Class to color mapping: {self.class_colors}")

def main():
    parser = argparse.ArgumentParser(description='Convert multi-class mask images to YOLO segmentation format.')
    parser.add_argument('input_directory_path', type=str, help='Path to the directory containing mask images.')
    parser.add_argument('output_directory_path', type=str, help='Path to save the annotation files.')
    args = parser.parse_args()

    converter = MultiClassMaskToYOLOConverter(args.input_directory_path, args.output_directory_path)
    converter.convert_directory_masks()

if __name__ == "__main__":
    main()