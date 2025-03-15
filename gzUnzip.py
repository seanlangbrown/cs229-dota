"""
Citation: Anthropic. (2024). Claude 3.5 Sonnet [Large Language Model]. Retrieved from https://www.anthropic.com
"""

import boto3
import gzip
import io
import json
import logging
import os
import sys
import time
from typing import Generator, List, Dict, Optional, Set
from botocore.exceptions import ClientError
from botocore.config import Config

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages upload checkpoints with constant file size."""
    
    def __init__(self, checkpoint_file: str, recent_etags_count: int = 5):
        self.checkpoint_file = checkpoint_file
        self.upload_id = None
        self.last_part_number = 0
        self.recent_etags = []
        self.recent_etags_count = recent_etags_count
        
    def load(self) -> bool:
        """Load checkpoint data if it exists."""
        if not os.path.exists(self.checkpoint_file):
            return False
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.upload_id = data.get('upload_id')
                self.last_part_number = data.get('last_part_number', 0)
                self.recent_etags = data.get('recent_etags', [])
                return bool(self.upload_id)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return False
    
    def save(self) -> bool:
        """Save checkpoint data to file."""
        try:
            checkpoint_dir = os.path.dirname(self.checkpoint_file)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            # Write to temporary file first to avoid corruption if interrupted
            temp_file = f"{self.checkpoint_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump({
                    'upload_id': self.upload_id,
                    'last_part_number': self.last_part_number,
                    'recent_etags': self.recent_etags
                }, f)
                
            # Atomic rename to ensure consistent checkpoint file
            os.replace(temp_file, self.checkpoint_file)
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            return False
    
    def update(self, upload_id: str, part_number: int, etag: str) -> None:
        """Update checkpoint with a newly completed part."""
        if not self.upload_id:
            self.upload_id = upload_id
        
        self.last_part_number = part_number
        
        # Keep only the most recent ETags
        self.recent_etags.append({'PartNumber': part_number, 'ETag': etag})
        if len(self.recent_etags) > self.recent_etags_count:
            self.recent_etags = self.recent_etags[-self.recent_etags_count:]
            
        self.save()
    
    def clear(self) -> None:
        """Clear checkpoint after successful completion."""
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
            except Exception as e:
                logger.error(f"Failed to remove checkpoint file: {str(e)}")

class S3TransferProcessor:
    def __init__(
        self,
        dest_bucket: str,
        dest_key: str,
        checkpoint_file: str,
        source_file_path: Optional[str] = None,
        source_bucket: Optional[str] = None,
        source_key: Optional[str] = None,
        part_size_mb: int = 100,
        max_retries: int = 5,
        retry_mode: str = 'adaptive'
    ):
        """
        Initialize S3 Transfer Processor.
        
        Args:
            dest_bucket: Destination S3 bucket name
            dest_key: Destination file key (JSON file)
            checkpoint_file: Path to checkpoint file for resuming
            source_file_path: Path to local gzipped file (optional)
            source_bucket: Source S3 bucket name (optional)
            source_key: Source file key (gzipped file) (optional)
            part_size_mb: Size of each upload part in MB
            max_retries: Maximum number of retries for failed operations
            retry_mode: Retry mode for S3 operations ('legacy', 'standard', or 'adaptive')
        """
        self.dest_bucket = dest_bucket
        self.dest_key = dest_key
        self.source_file_path = source_file_path
        self.source_bucket = source_bucket
        self.source_key = source_key
        self.part_size = part_size_mb * 1024 * 1024  # Convert MB to bytes
        self.max_retries = max_retries
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_file)
        
        # Configure S3 client with retry settings
        boto_config = Config(
            retries={
                'max_attempts': max_retries,
                'mode': retry_mode
            }
        )
        self.s3_client = boto3.client('s3', config=boto_config)
        
    def stream_gzip_from_s3(self, start_part: int = 1) -> Generator[bytes, None, None]:
        """Stream and decompress gzipped content from S3 as raw bytes."""
        try:
            logger.info(f"Starting to stream from s3://{self.source_bucket}/{self.source_key}")
            
            # Get the gzipped object from S3
            response = self.s3_client.get_object(
                Bucket=self.source_bucket,
                Key=self.source_key
            )
            
            # Create a gzip reader around the StreamingBody
            with gzip.GzipFile(fileobj=response['Body'], mode='rb') as gz:
                current_part = 1
                
                # Skip already processed parts if resuming
                bytes_to_skip = (start_part - 1) * self.part_size
                if bytes_to_skip > 0:
                    logger.info(f"Skipping {bytes_to_skip / (1024 * 1024):.2f}MB to resume from part {start_part}")
                    gz.read(bytes_to_skip)
                    current_part = start_part
                
                # Process remaining data
                while True:
                    chunk = gz.read(self.part_size)
                    if not chunk:
                        break
                    yield chunk
                    current_part += 1
                    
        except ClientError as e:
            logger.error(f"Error streaming from S3: {str(e)}")
            raise

    def stream_gzip_from_local_disk(self, start_part: int = 1) -> Generator[bytes, None, None]:
        """Read and decompress gzipped content from local disk as raw bytes."""
        try:
            logger.info(f"Starting to stream from local file: {self.source_file_path}")
            
            # Open the file and decompress it
            with open(self.source_file_path, 'rb') as f:
                with gzip.GzipFile(fileobj=f, mode='rb') as gz:
                    current_part = 1
                    
                    # Skip already processed parts if resuming
                    bytes_to_skip = (start_part - 1) * self.part_size
                    if bytes_to_skip > 0:
                        logger.info(f"Skipping {bytes_to_skip / (1024 * 1024):.2f}MB to resume from part {start_part}")
                        gz.read(bytes_to_skip)
                        current_part = start_part
                    
                    # Process remaining data
                    while True:
                        chunk = gz.read(self.part_size)
                        if not chunk:
                            break
                        yield chunk
                        current_part += 1
                    
        except Exception as e:
            logger.error(f"Error reading from local file: {str(e)}")
            raise
            
    def upload_part(
        self, 
        upload_id: str, 
        part_number: int, 
        part_data: bytes
    ) -> Dict:
        """
        Upload a single part with retry logic.
        
        Returns:
            Dict containing PartNumber and ETag
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                response = self.s3_client.upload_part(
                    Bucket=self.dest_bucket,
                    Key=self.dest_key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=part_data
                )
                
                logger.info(f"Uploaded part {part_number}")
                
                # Update checkpoint
                self.checkpoint_manager.update(upload_id, part_number, response['ETag'])
                
                return {
                    'PartNumber': part_number,
                    'ETag': response['ETag']
                }
                
            except ClientError as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Failed to upload part {part_number} after {self.max_retries} retries: {str(e)}")
                    raise
                
                # Exponential backoff
                wait_time = 2 ** retries
                logger.warning(f"Upload part {part_number} failed. Retrying in {wait_time}s... ({retries}/{self.max_retries})")
                time.sleep(wait_time)
    
    def get_uploaded_parts(self, upload_id: str) -> List[Dict]:
        """
        Get a list of all parts already uploaded to S3.
        
        Returns:
            List of parts (PartNumber and ETag)
        """
        parts = []
        next_part_marker = None
        
        while True:
            # Parameters for list_parts API call
            params = {
                'Bucket': self.dest_bucket,
                'Key': self.dest_key,
                'UploadId': upload_id,
                'MaxParts': 1000
            }
            
            # Add part marker for pagination if we have one
            if next_part_marker:
                params['PartNumberMarker'] = next_part_marker
                
            # Get batch of parts
            response = self.s3_client.list_parts(**params)
            
            # Add parts to our list
            if 'Parts' in response:
                for part in response['Parts']:
                    parts.append({
                        'PartNumber': part['PartNumber'],
                        'ETag': part['ETag']
                    })
            
            # Check if there are more parts to fetch
            if response['IsTruncated']:
                next_part_marker = response['NextPartNumberMarker']
            else:
                break
                
        return sorted(parts, key=lambda x: x['PartNumber'])
    
    def upload_to_s3(self, data_generator: Generator[bytes, None, None], start_part: int = 1) -> bool:
        """
        Stream and upload data to S3 using multipart upload.
        
        Args:
            data_generator: Generator yielding data chunks
            start_part: Part number to start uploading from (for resuming)
            
        Returns:
            bool: True if successful, False otherwise
        """
        upload_id = self.checkpoint_manager.upload_id
        
        # Start new upload if not resuming
        if not upload_id:
            try:
                logger.info(f"Starting new multipart upload to s3://{self.dest_bucket}/{self.dest_key}")
                mpu = self.s3_client.create_multipart_upload(
                    Bucket=self.dest_bucket,
                    Key=self.dest_key
                )
                upload_id = mpu['UploadId']
                self.checkpoint_manager.upload_id = upload_id
                self.checkpoint_manager.save()
            except Exception as e:
                logger.error(f"Error initializing upload: {str(e)}")
                return False
        else:
            logger.info(f"Resuming multipart upload with ID {upload_id} to s3://{self.dest_bucket}/{self.dest_key}")
            logger.info(f"Already completed {start_part - 1} parts")
        
        try:
            part_number = start_part
            
            # Process data chunks
            for chunk in data_generator:
                if not chunk:  # Skip empty chunks
                    continue
                
                # Upload part
                self.upload_part(upload_id, part_number, chunk)
                part_number += 1
            
            # Get a list of all parts from S3
            logger.info("Retrieving completed parts from S3...")
            completed_parts = self.get_uploaded_parts(upload_id)
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.dest_bucket,
                Key=self.dest_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': completed_parts}
            )
            
            logger.info(f"Multipart upload completed successfully with {len(completed_parts)} parts")
            
            # Clear checkpoint
            self.checkpoint_manager.clear()
            return True
                
        except Exception as e:
            logger.error(f"Error during upload: {str(e)}")
            # Don't abort the upload - let the checkpoint system handle resuming
            return False
    
    def process(self) -> bool:
        """
        Execute the full processing pipeline with checkpoint support.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load checkpoint if available
            resume = self.checkpoint_manager.load()
            start_part = 1
            
            if resume:
                # If resuming, start from the part after the last successful one
                start_part = self.checkpoint_manager.last_part_number + 1
                logger.info(f"Resuming from part {start_part}")
            
            # Get data generator
            data_generator = None
            if self.source_file_path:
                data_generator = self.stream_gzip_from_local_disk(start_part)
            elif self.source_bucket and self.source_key:
                data_generator = self.stream_gzip_from_s3(start_part)
            else:
                raise ValueError("Either source_file_path or (source_bucket and source_key) must be provided")
            
            # Upload to S3
            success = self.upload_to_s3(data_generator, start_part)
            
            if success:
                logger.info(f"Processing completed successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return False

def main():
    # Example usage with source in local file
    processor = S3TransferProcessor(
        source_file_path="./downloads/yasp-dump.json.gz",
        dest_bucket='cs229-dota',
        dest_key='data/yasp-dump-90gb.json',
        checkpoint_file="./upload_checkpoint.json",
        part_size_mb=600,
        max_retries=10
    )
    
    success = processor.process()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()