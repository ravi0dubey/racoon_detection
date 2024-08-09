import argparse
import multiprocessing
import cv2
import logging
from google.cloud import storage
from pathlib import Path
import os
import tempfile

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_storage_client():
    """Gets the Google Cloud Storage client, using default credentials if service account json is not available."""
    service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service_account.json')
    if os.path.exists(service_account_path):
        return storage.Client.from_service_account_json(service_account_path)
    else:
        return storage.Client()

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    logger.debug(f"Downloading blob: bucket_name={bucket_name}, source_blob_name={source_blob_name}, destination_file_name={destination_file_name}")
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    logger.debug(f"Uploading blob: bucket_name={bucket_name}, source_file_name={source_file_name}, destination_blob_name={destination_blob_name}")
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(f"File {source_file_name} uploaded to {destination_blob_name}.")

def extract_frames(video_file, output_dir, frame_rate, input_bucket=None, output_bucket=None):
    try:
        logger.debug(f"Starting frame extraction: video_file={video_file}, output_dir={output_dir}, frame_rate={frame_rate}, input_bucket={input_bucket}, output_bucket={output_bucket}")
        local_video_path = video_file
        if input_bucket:
            local_video_path = os.path.join("/tmp", Path(video_file).name)
            download_blob(input_bucket, video_file, local_video_path)

        logger.info(f"Processing video file: {local_video_path}")
        cap = cv2.VideoCapture(str(local_video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {local_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(round(fps / frame_rate))
        success, image = cap.read()
        frame_count = 0
        image_count = 0

        logger.debug(f"Video properties: total_frames={total_frames}, fps={fps}, interval={interval}")

        with tempfile.TemporaryDirectory() as temp_dir:
            while success:
                if frame_count % interval == 0:
                    output_filename = f"{Path(video_file).stem}_{image_count+1:05d}.jpg"
                    local_output_path = os.path.join(temp_dir, output_filename)
                    cv2.imwrite(local_output_path, image)
                    logger.debug(f"Saved frame: {local_output_path}")
                    if output_bucket:
                        upload_blob(output_bucket, local_output_path, os.path.join(output_dir, output_filename))
                    else:
                        os.rename(local_output_path, os.path.join(output_dir, output_filename))
                    image_count += 1
                success, image = cap.read()
                frame_count += 1

        cap.release()
        if input_bucket:
            os.remove(local_video_path)
        logger.info(f"Successfully extracted {image_count} frames from {video_file}")
        return True, f"Successfully extracted {image_count} frames from {video_file}"
    except Exception as e:
        logger.error(f"Error processing {video_file}: {str(e)}")
        return False, f"Error processing {video_file}: {str(e)}"

def process_video(video_file, output_dir, frame_rate, input_bucket=None, output_bucket=None):
    """
    Process a single video file by extracting frames and logging the result.

    Args:
        video_file (str): Path to the video file in GCS or local.
        output_dir (str): Directory to save the extracted frames.
        frame_rate (int): Rate at which frames should be extracted.
        input_bucket (str, optional): GCS input bucket name.
        output_bucket (str, optional): GCS output bucket name.
    """
    logger.debug(f"Processing video: video_file={video_file}, output_dir={output_dir}, frame_rate={frame_rate}, input_bucket={input_bucket}, output_bucket={output_bucket}")
    filename = Path(video_file).name
    result, message = extract_frames(video_file, output_dir, frame_rate, input_bucket, output_bucket)
    if result:
        logger.info(f"Processed {filename}: {message}")
    else:
        logger.error(f"Processed {filename}: {message}")

def main(input_source, output_path, frame_rate):
    logger.info(f"Received arguments: input_source={input_source}, output_path={output_path}, frame_rate={frame_rate}")
    
    input_bucket = None
    output_bucket = None
    
    if input_source.startswith("gs://"):
        input_bucket = input_source[5:].split('/')[0]
        input_prefix = '/'.join(input_source[5:].split('/')[1:])
        storage_client = get_storage_client()
        blobs = storage_client.list_blobs(input_bucket, prefix=input_prefix)
        video_files = [blob.name for blob in blobs if blob.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    else:
        video_files = [str(f) for f in Path(input_source).glob('**/*') if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')]

    if output_path.startswith("gs://"):
        output_bucket = output_path[5:].split('/')[0]
        output_prefix = '/'.join(output_path[5:].split('/')[1:])
    else:
        output_prefix = output_path
        Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.debug(f"Video files to process: {video_files}")

    with multiprocessing.Pool() as pool:
        pool.starmap(process_video, [(video, output_prefix, frame_rate, input_bucket, output_bucket) for video in video_files])

if __name__ == "__main__":
    input_source = os.environ.get('INPUT_SOURCE')
    output_path = os.environ.get('OUTPUT_PATH')
    frame_rate = int(os.environ.get('FRAME_RATE', 1))
    
    if not input_source or not output_path:
        logger.error("INPUT_SOURCE and OUTPUT_PATH must be set in the environment variables.")
        exit(1)
    
    main(input_source, output_path, frame_rate)
