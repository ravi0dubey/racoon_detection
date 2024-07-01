import argparse
import multiprocessing
from pathlib import Path
import cv2

def extract_frames(video_file, output_dir, frame_rate):
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
    filename = video_file.name
    result, message = extract_frames(video_file, output_dir, frame_rate)
    print(f"Processed {filename}: {message}")

def main(root_dir, output_dir, frame_rate):
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
    parser.add_argument("root_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--frame_rate", type=int, default=1)
    args = parser.parse_args()
    main(args.root_dir, args.output_dir, args.frame_rate)
