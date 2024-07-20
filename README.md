# racoon_detection
## Overview
The eh_video_to_image_extractor.py script extracts frames from video files and saves them as images. It processes all video files (with extensions .mp4, .avi, .mov, .mkv) found recursively in a specified root directory. Frames are extracted at a specified frame rate and saved in a designated output directory.

```
python eh_video_to_image_extractor.py "C:\\Users\\Ravi0dubey\\Videos" "C:\\Users\\Ravi0dubey\\Videos\\output" [--frame_rate 2]
```

### Building the Docker Image Locally
Ensure Docker is installed and running on your local machine.
```
cd /path/to/your/project
docker build -t video-frame-extractor:latest .
```
### Running the Docker Container Locally
```
mkdir -p "C:\\Users\\Ravi0dubey\\Videos" "C:\\Users\\Ravi0dubey\\Videos\\output"
# Copy some video files to the /tmp/input directory for testing.

For local paths:

# docker run --rm -it \
  -v "C:\\Users\\Ravi0dubey\\Videos\\Bandicam":/app/input \
  -v "C:\\Users\\Ravi0dubey\\Videos\\bandicam_output":/app/output \
  video-frame-extractor:latest /app/input /app/output --frame_rate 2
# For GCS paths:
docker run --rm -it \
  -v /path/to/gcp_service_account.json:/app/gcp_service_account.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp_service_account.json \
  video-frame-extractor:latest gs://input-bucket gs://output-bucket --frame_rate 2

```