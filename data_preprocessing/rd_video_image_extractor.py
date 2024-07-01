"""
Script to extract frames from video files and save them as images.

The script processes all video files (mp4, avi, mov, mkv) found recursively
in the specified root directory (`root_dir`). It extracts frames at a specified
frame rate (`frame_rate`) and saves them in the `output_dir`.

Usage:
    python script_name.py root_dir output_dir [--frame_rate FRAME_RATE]

Example:
    python extract_frames.py /path/to/videos /path/to/output --frame_rate 2
"""

import argparse
import multiprocessing
from pathlib import Path
import cv2

def extract_frames(video_file, output_dir, frame_rate):
    """
    Extract frames from a video file.

    Args:
        video_file (Path): Path to the video file.
        output_dir (Path): Directory to save the extracted frames.
        frame_rate (int): Rate at which frames should be extracted.

    Returns:
        tuple: (success (bool), message (str))
            success: True if frames are successfully extracted, False otherwise.
            message: Description of the result.
    """
    try:
        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)    
        interval = int(round(fps / frame_rate))       
        success, image = cap.read()
        frame_count = 0
        image_count = 0
        while success:
            if frame_count % interval == 0:
                output_filename = f"{video_file.stem}_{image_count+1:03d}.jpg"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), image)
                image_count += 1
            success, image = cap.read()
            frame_count += 1
        
        cap.release()
        return True, f"Successfully extracted {image_count} frames from {video_file}"    
    except Exception as e:
        return False, f"Error processing {video_file}: {str(e)}"

def process_video(video_file, output_dir, frame_rate):
    """
    Process a single video file by extracting frames and printing the result.

    Args:
        video_file (Path): Path to the video file.
        output_dir (Path): Directory to save the extracted frames.
        frame_rate (int): Rate at which frames should be extracted.
    """
    filename = video_file.name
    result, message = extract_frames(video_file, output_dir, frame_rate)
    print(f"Processed {filename}: {message}")

def main(root_dir, output_dir, frame_rate):
    """
    Main function to process all video files in the specified directory.

    Args:
        root_dir (str): Root directory containing video files.
        output_dir (str): Directory to save the extracted frames.
        frame_rate (int): Rate at which frames should be extracted.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)   
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = list(root_path.glob('**/*.mp4')) + \
                  list(root_path.glob('**/*.avi')) + \
                  list(root_path.glob('**/*.mov')) + \
                  list(root_path.glob('**/*.mkv'))
    with multiprocessing.Pool() as pool:
        pool.starmap(process_video, [(video, output_path, frame_rate) for video in video_files])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="Root directory containing video files")
    parser.add_argument("output_dir", type=str, help="Directory to save extracted frames")
    parser.add_argument("--frame_rate", type=int, default=1, help="Frame extraction rate (frames per second)")
    args = parser.parse_args()
    main(args.root_dir, args.output_dir, args.frame_rate)