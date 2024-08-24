import os
import fiftyone as fo
import fiftyone.brain as fob
from google.cloud import storage
import logging
import argparse
import sys
from approx_dups import find_approximate_duplicates, deduplicate_approximate_duplicates

class GCSImageHandler:
    """
    Class to handle downloading images from Google Cloud Storage, creating a FiftyOne dataset,
    and uploading results back to Google Cloud Storage.
    """

    def __init__(self, input_bucket, output_bucket, annotation_set_bucket, images_prefix, local_dir, credentials_path):
        """
        Initializes the handler with GCS bucket details and local storage path.
        
        :param input_bucket: Name of the GCS input bucket.
        :param output_bucket: Name of the GCS output bucket.
        :param annotation_set_bucket: Name of the GCS annotation set bucket.
        :param images_prefix: Prefix to filter images in the input bucket.
        :param local_dir: Local directory to store downloaded images.
        :param credentials_path: Path to the service account credentials.
        """
        self.input_bucket_name = input_bucket
        self.output_bucket_name = output_bucket
        self.annotation_set_bucket_name = annotation_set_bucket
        self.images_prefix = images_prefix
        self.local_dir = local_dir
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', credentials_path)
        if os.path.exists(service_account_path):
            storage.Client.from_service_account_json(service_account_path)
        else:
            storage.Client()

        self.client = storage.Client()
        self.input_bucket = self.client.get_bucket(self.input_bucket_name)
        self.output_bucket = self.client.get_bucket(self.output_bucket_name)
        self.annotation_set_bucket = self.client.get_bucket(self.annotation_set_bucket_name)
        os.makedirs(self.local_dir, exist_ok=True)

    def download_images(self):
        """
        Downloads images containing '_capture_' from the specified GCS bucket to the local directory.
        Skips downloading if the image already exists locally.
        """
        blobs = self.input_bucket.list_blobs(prefix=self.images_prefix)
        logging.info(f"Downloading images from {self.input_bucket_name}/{self.images_prefix} to {self.local_dir}")
        for blob in blobs:
            if blob.name.endswith(".jpg"):
                local_path = os.path.join(self.local_dir, os.path.basename(blob.name))
                if not os.path.exists(local_path):
                    blob.download_to_filename(local_path)
                    logging.info(f"Downloaded {blob.name} to {local_path}")
                else:
                    logging.info(f"Skipped {blob.name}, already exists at {local_path}")

    def _get_next_annotation_dir(self):
        """
        Determines the next logical directory number in the annotation set bucket.
        """
        # List all blobs in the bucket
        blobs = list(self.annotation_set_bucket.list_blobs(delimiter='/'))
        
        # Extract directory names (ignoring files in the root)
        dir_prefixes = set()
        for blob in blobs:
            if blob.name.count('/') > 0:  # This ensures we're looking at directories, not root-level files
                dir_prefix = blob.name.split('/')[0]
                if dir_prefix.isdigit():
                    dir_prefixes.add(dir_prefix)

        # Find the highest existing directory number
        existing_nums = [int(dir) for dir in dir_prefixes]
        next_num = max(existing_nums, default=0) + 1
        
        return f"{next_num:03d}"

    def upload_images(self, dataset):
        """
        Uploads the processed dataset images to the specified GCS output bucket and annotation set bucket.
        Only uploads files that are not already in the output bucket.
        
        :param dataset: FiftyOne dataset containing the samples.
        """
        # Get list of existing files in output bucket
        existing_files = set(blob.name for blob in self.output_bucket.list_blobs())

        # Find files to upload (delta)
        files_to_upload = []
        for sample in dataset:
            local_path = sample.filepath
            blob_path = os.path.relpath(local_path, self.local_dir)
            if blob_path not in existing_files:
                files_to_upload.append((local_path, blob_path))

        # Upload delta files to output bucket
        for local_path, blob_path in files_to_upload:
            blob = self.output_bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            logging.info(f"Uploaded {local_path} to {self.output_bucket_name}/{blob_path}")

        # Upload delta files to annotation set bucket
        if files_to_upload:
            next_dir = self._get_next_annotation_dir()
            for local_path, blob_path in files_to_upload:
                annotation_blob_path = f"{next_dir}/{blob_path}"
                blob = self.annotation_set_bucket.blob(annotation_blob_path)
                blob.upload_from_filename(local_path)
                logging.info(f"Uploaded {local_path} to {self.annotation_set_bucket_name}/{annotation_blob_path}")

class FiftyOneDatasetCreator:
    """
    Class to handle creating a FiftyOne dataset from local images.
    """

    def __init__(self, dataset_name, local_dir, threshold):
        """
        Initializes the dataset creator with the dataset name, local directory containing images, and similarity threshold.
        
        :param dataset_name: Name of the FiftyOne dataset.
        :param local_dir: Local directory containing images.
        :param threshold: Threshold for finding approximate duplicates.
        """
        self.dataset_name = dataset_name
        self.local_dir = local_dir
        self.threshold = threshold
        self.dataset = None

    def delete_existing_dataset(self):
        """
        Deletes the existing FiftyOne dataset if it exists.
        """
        if fo.dataset_exists(self.dataset_name):
            fo.delete_dataset(self.dataset_name)
            logging.info(f"Deleted existing dataset {self.dataset_name}")

    def create_dataset(self):
        """
        Creates a new FiftyOne dataset.
        """
        self.dataset = fo.Dataset(self.dataset_name)
        logging.info(f"Created new dataset {self.dataset_name}")

    def add_samples_to_dataset(self):
        """
        Adds images from the local directory to the FiftyOne dataset.
        """
        for root, _, files in os.walk(self.local_dir):
            for file in files:
                if file.endswith(".jpg"):
                    sample = fo.Sample(filepath=os.path.join(root, file))
                    self.dataset.add_sample(sample)
                    logging.info(f"Added {file} to the dataset")

    def save_dataset(self):
        """
        Saves the FiftyOne dataset.
        """
        self.dataset.save()
        logging.info(f"Dataset {self.dataset_name} saved")
    
    def compute_similarity(self):
        """
        Computes the similarity between samples in the dataset.
        """
        fob.compute_similarity(self.dataset, brain_key="sim", metric="cosine")
        logging.info("Computed similarity between samples")

    def find_approximate_duplicates(self):
        """
        Finds and prints the approximate duplicates.
        """
        response = find_approximate_duplicates(self.dataset, brain_key="sim", threshold=self.threshold)
        logging.info(f"Found approximate duplicates: {response}")
        return response

    def deduplicate_approximate_duplicates(self):
        """
        Deletes all approximate duplicates from the dataset.
        """
        deduplicate_approximate_duplicates(self.dataset)
        logging.info("Deleted all approximate duplicates")

def main(input_bucket, output_bucket, annotation_set_bucket, images_prefix, local_dir, credentials_path, threshold):
    """
    Main function to execute the downloading and dataset creation process.
    
    :param input_bucket: Name of the GCS input bucket.
    :param output_bucket: Name of the GCS output bucket.
    :param annotation_set_bucket: Name of the GCS annotation set bucket.
    :param images_prefix: Prefix to filter images in the input bucket.
    :param local_dir: Local directory to store downloaded images.
    :param credentials_path: Path to the service account credentials.
    :param threshold: Threshold for finding approximate duplicates.
    """
    logging.basicConfig(level=logging.INFO)
    if images_prefix == "default_prefix":
        images_prefix = ""

    gcs_handler = GCSImageHandler(input_bucket, output_bucket, annotation_set_bucket, images_prefix, local_dir, credentials_path)
    gcs_handler.download_images()

    dataset_creator = FiftyOneDatasetCreator("rgb_pingthru_dataset", local_dir, threshold)
    dataset_creator.delete_existing_dataset()
    dataset_creator.create_dataset()
    dataset_creator.add_samples_to_dataset()
    dataset_creator.save_dataset()
    dataset_creator.compute_similarity()
    approximate_duplicates = dataset_creator.find_approximate_duplicates()
    dataset_creator.deduplicate_approximate_duplicates()
    dataset_creator.save_dataset()

    gcs_handler.upload_images(dataset_creator.dataset)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Download images from GCS and create a FiftyOne dataset.")
#     parser.add_argument("--input_bucket", type=str, default="02-extracted-images-4v6cnheu", help="Input Google Cloud Storage bucket name")
#     parser.add_argument("--output_bucket", type=str, default="03-dedup-images-4v6cnheu", help="Output Google Cloud Storage bucket name")
#     parser.add_argument("--annotation_set_bucket", type=str, default="04_cvat_annotations_4v6cnheu", help="Annotation set Google Cloud Storage bucket name")
#     parser.add_argument("--images_prefix", type=str, default="", help="Prefix to filter images in the bucket")
#     parser.add_argument("--local_dir", type=str, default="/app/images", help="Local directory to store downloaded images")
#     parser.add_argument("--credentials_path", type=str, default="/app/service_account.json", help="Path to the service account credentials")
#     # parser.add_argument("--local_dir", type=str, default="D:\\Mentoring_Project\\racoon_project\\data_preprocessing\\eh_unique_frame_extraction\\images", help="Local directory to store downloaded images")
#     # parser.add_argument("--credentials_path", type=str, default="D:\\Mentoring_Project\\racoon_project\\data_preprocessing\\eh_unique_frame_extraction\\service_account.json", help="Path to the service account credentials")
#     parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for finding approximate duplicates")

#     args = parser.parse_args()

#     main(args.input_bucket, args.output_bucket, args.annotation_set_bucket, args.images_prefix, args.local_dir, args.credentials_path, args.threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from GCS and create a FiftyOne dataset.")
    parser.add_argument("--input_bucket", type=str, default=os.getenv("INPUT_BUCKET", ""), help="Input Google Cloud Storage bucket name")
    parser.add_argument("--output_bucket", type=str, default=os.getenv("OUTPUT_BUCKET", ""), help="Output Google Cloud Storage bucket name")
    parser.add_argument("--annotation_set_bucket", type=str, default=os.getenv("ANNOTATION_SET_BUCKET", ""), help="Annotation set Google Cloud Storage bucket name")
    parser.add_argument("--images_prefix", type=str, default=os.getenv("IMAGES_PREFIX", ""), help="Prefix to filter images in the bucket")
    parser.add_argument("--local_dir", type=str, default=os.getenv("LOCAL_DIR", "/app/images"), help="Local directory to store downloaded images")
    parser.add_argument("--credentials_path", type=str, default=os.getenv("CREDENTIALS_PATH", "/app/service_account.json"), help="Path to the service account credentials")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("THRESHOLD", 0.3)), help="Threshold for finding approximate duplicates")

    args = parser.parse_args()

    main(args.input_bucket, args.output_bucket, args.annotation_set_bucket, args.images_prefix, args.local_dir, args.credentials_path, args.threshold)