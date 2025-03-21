"""
Citation: Anthropic. (2024). Claude 3.5 Sonnet [Large Language Model]. Retrieved from https://www.anthropic.com
"""

import boto3
import os
from typing import Dict, List
import math
import threading
import sys
from botocore.exceptions import ClientError
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3LargeUploader:
    def __init__(self, bucket: str, file_path: str, key: str, part_size_mb: int = 100):
        """
        Initialize uploader with configuration.
        
        Args:
            bucket: S3 bucket name
            file_path: Path to local file
            key: S3 key (destination path)
            part_size_mb: Size of each part in MB (default 100MB)
        """
        self.bucket = bucket
        self.file_path = file_path
        self.key = key
        self.part_size = part_size_mb * 1024 * 1024  # Convert MB to bytes
        self.s3_client = boto3.client('s3')
        self.upload_id = None
        self.parts: List[Dict] = []
        
        # Calculate total parts
        self.total_parts = math.ceil(os.path.getsize(file_path) / self.part_size)
        
        # Progress tracking
        self.uploaded_parts = 0
        self.lock = threading.Lock()

    def upload_part(self, part_number: int, start_byte: int) -> Dict:
        """
        Upload a single part of the file.
        
        Args:
            part_number: Part number (1-based)
            start_byte: Starting byte of this part
            
        Returns:
            Dict containing PartNumber and ETag
        """
        # Calculate end byte for this part
        end_byte = min(start_byte + self.part_size, os.path.getsize(self.file_path))
        
        try:
            # Read and upload the part
            with open(self.file_path, 'rb') as f:
                f.seek(start_byte)
                part_data = f.read(end_byte - start_byte)
                
                response = self.s3_client.upload_part(
                    Bucket=self.bucket,
                    Key=self.key,
                    PartNumber=part_number,
                    UploadId=self.upload_id,
                    Body=part_data
                )
            
            # Update progress
            with self.lock:
                self.uploaded_parts += 1
                progress = (self.uploaded_parts / self.total_parts) * 100
                sys.stdout.write(f'\rProgress: {progress:.2f}% ({self.uploaded_parts}/{self.total_parts} parts)')
                sys.stdout.flush()
            
            return {
                'PartNumber': part_number,
                'ETag': response['ETag']
            }
            
        except Exception as e:
            logger.error(f"Failed to upload part {part_number}: {str(e)}")
            raise

    def upload(self, max_workers: int = 10) -> bool:
        """
        Upload the file using multipart upload.
        
        Args:
            max_workers: Maximum number of concurrent upload threads
            
        Returns:
            bool: True if upload was successful
        """
        try:
            # Initiate multipart upload
            logger.info(f"Starting upload of {self.file_path} to s3://{self.bucket}/{self.key}")
            response = self.s3_client.create_multipart_upload(
                Bucket=self.bucket,
                Key=self.key
            )
            self.upload_id = response['UploadId']
            
            # Prepare parts for concurrent upload
            upload_parts = []
            for part_number in range(1, self.total_parts + 1):
                start_byte = (part_number - 1) * self.part_size
                upload_parts.append((part_number, start_byte))
            
            # Upload parts concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.upload_part, part_number, start_byte)
                    for part_number, start_byte in upload_parts
                ]
                
                # Collect results
                self.parts = [future.result() for future in futures]
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=self.key,
                UploadId=self.upload_id,
                MultipartUpload={'Parts': sorted(self.parts, key=lambda x: x['PartNumber'])}
            )
            
            logger.info(f"\nUpload completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            if self.upload_id:
                logger.info("Aborting multipart upload...")
                self.s3_client.abort_multipart_upload(
                    Bucket=self.bucket,
                    Key=self.key,
                    UploadId=self.upload_id
                )
            return False

def main():
    # Example usage
    # uploader = S3LargeUploader(
    #     bucket='cs229-dota',
    #     file_path='/Users/slangbro/Downloads/yasp-dump-2015-12-18.json.gz',
    #     key='downloads/yasp-dump-2015-12-18.json.gz',
    #     part_size_mb=500  # Adjust based on your needs and memory constraints
    # )

    uploader = S3LargeUploader(
        bucket='cs229-dota',
        file_path='./downloads/yasp-dump.json.gz',
        key='downloads/yasp-dump-90gb.json.gz',
        part_size_mb=500  # Adjust based on your needs and memory constraints
    )
    
    success = uploader.upload(max_workers=5)  # Adjust number of workers based on your needs
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()