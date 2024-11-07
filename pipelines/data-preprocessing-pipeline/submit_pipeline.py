from google.cloud import aiplatform
from google.cloud import storage
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = 'racoon-detection-427421'
REGION = 'us-central1'

def validate_gcs_path(gcs_path: str, check_write_access: bool = False) -> bool:
    """Validates GCS path and checks permissions."""
    try:
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        bucket_name = gcs_path.split('/')[2]
        prefix = '/'.join(gcs_path.split('/')[3:])
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Check if bucket exists
        if not bucket.exists():
            raise ValueError(f"Bucket does not exist: {bucket_name}")
            
        # Check read access
        try:
            next(bucket.list_blobs(prefix=prefix, max_results=1))
        except StopIteration:
            logger.warning(f"No objects found in {gcs_path}")
        
        # Check write access if required
        if check_write_access:
            test_blob = bucket.blob(f"{prefix}/test_write_access")
            test_blob.upload_from_string("")
            test_blob.delete()
            
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False

def submit_pipeline(input_source: str, output_path: str, frame_rate: int = 1):
    """Submits and monitors the video frame extraction pipeline."""
    try:
        # Validate inputs
        if not validate_gcs_path(input_source):
            raise ValueError("Invalid or inaccessible input source")
        if not validate_gcs_path(output_path, check_write_access=True):
            raise ValueError("Invalid or inaccessible output path")
        if frame_rate < 1:
            raise ValueError("Frame rate must be positive")

        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)

        # Create and run pipeline job
        pipeline_job = aiplatform.PipelineJob(
            display_name='video_frame_extraction_pipeline',
            template_path='video_frame_extraction_pipeline.json',
            parameter_values={
                'input_source': input_source,
                'output_path': output_path,
                'frame_rate': frame_rate
            },
            enable_caching=False
        )

        logger.info(f"Submitting pipeline job with parameters:\n"
                   f"Input Source: {input_source}\n"
                   f"Output Path: {output_path}\n"
                   f"Frame Rate: {frame_rate}")
        
        pipeline_job.run(sync=True)
        
        # Check pipeline status
        if pipeline_job.state.name == "PIPELINE_STATE_SUCCEEDED":
            logger.info("Pipeline completed successfully")
        else:
            logger.error(f"Pipeline ended in state: {pipeline_job.state.name}")
            
    except Exception as e:
        logger.error(f"Pipeline submission failed: {str(e)}")
        raise

def main():
    try:
        # You could add argparse here for command-line arguments
        input_source = 'gs://01-raw_dataset-4v6cnheu'
        output_path = 'gs://racoon-temp-bucket'
        frame_rate = 1

        submit_pipeline(input_source, output_path, frame_rate)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()