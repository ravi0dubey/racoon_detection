terraform {
  required_providers {
    google = {
      source  = "hashicorp/google" # its google as we are working on gcp
      version = "4.51.0"          # replace with aws or azure if needed 
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
  
  backend "gcs" {
    bucket  = "terraform-state-eh-ml-mentorship-4" # Replace with your bucket name
    prefix  = "terraform/state"
  }
}

provider "google" {
  project = "racoon-detection-427421"  # Replace with your actual project ID
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# VPC Network
resource "google_compute_network" "vpc_network" {
  name                    = "terraform-network"
  auto_create_subnetworks = false

  lifecycle {
    ignore_changes = [name]  # Ignore changes to the network name
  }
}

# Storage Buckets (applying the lifecycle pattern)
resource "google_storage_bucket" "raw_dataset" {
  name          = "01-raw_dataset-${random_string.bucket_suffix.result}"
  location      = "us-east1"
  force_destroy = true
  uniform_bucket_level_access = true

  lifecycle {
    ignore_changes = [name]
  }

  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "extracted_images" {
  name          = "02-extracted-images-${random_string.bucket_suffix.result}"
  location      = "us-east1"
  force_destroy = true
  uniform_bucket_level_access = true

  lifecycle {
    ignore_changes = [name]
  }

  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "dedup_images" {
  name          = "03-dedup-images-${random_string.bucket_suffix.result}"
  location      = "us-east1"
  force_destroy = true
  uniform_bucket_level_access = true

  lifecycle {
    ignore_changes = [name]
  }

  versioning {
    enabled = true
  }
}
# resource "google_storage_bucket" "extracted_images" {
#   name          = "02-cloud-build-log-bucketd-images-${random_string.bucket_suffix.result}"
#   location      = "us-east1"
#   force_destroy = true
#   uniform_bucket_level_access = true

#   lifecycle {
#     ignore_changes = [name]
#   }

#   versioning {
#     enabled = true
#   }
# }