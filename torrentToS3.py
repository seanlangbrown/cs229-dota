"""
Citation: Anthropic. (2024). Claude 3.5 Sonnet [Large Language Model]. Retrieved from https://www.anthropic.com
"""

import libtorrent as lt
import boto3
import time
import os
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import gzip

import requests
import tempfile
from urllib.parse import urlparse

def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

class TorrentS3Service:
    def __init__(
        self,
        torrent_path: str,  # Can be local path or HTTP URL
        download_path: str,
        s3_bucket: str,
        s3_key: str,
        log_file: str,
        max_retry_hours: int = 72,  # Maximum time to keep retrying
        check_interval: int = 300,  # Check progress every 5 minutes
        part_size_mb: int = 100
    ):
        self.torrent_path = torrent_path
        self.download_path = download_path
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.max_retry_hours = max_retry_hours
        self.check_interval = check_interval
        self.part_size = part_size_mb * 1024 * 1024
        
        # Set up logging
        global logger
        logger = setup_logging(log_file)
        
        # Initialize AWS client
        self.s3_client = boto3.client('s3')
        
        # Ensure download directory exists
        os.makedirs(download_path, exist_ok=True)

    def get_torrent_file(self) -> str:
        """Get the torrent file, downloading it first if it's a URL."""
        if self.torrent_path.startswith(('http://', 'https://')):
            try:
                logger.info(f"Downloading torrent file from {self.torrent_path}")
                response = requests.get(self.torrent_path, timeout=30)
                response.raise_for_status()
                
                # Create a temporary file for the torrent
                with tempfile.NamedTemporaryFile(delete=False, suffix='.torrent') as tf:
                    tf.write(response.content)
                    temp_path = tf.name
                
                logger.info(f"Torrent file downloaded to {temp_path}")
                return temp_path
            except Exception as e:
                logger.error(f"Failed to download torrent file: {str(e)}")
                raise
        return self.torrent_path

    def wait_for_torrent_download(self) -> Optional[str]:
        """
        Download torrent and wait indefinitely if no peers are available.
        Returns the path to the downloaded file or None if max retry time exceeded.
        """
        # Get local path to torrent file
        torrent_file_path = self.get_torrent_file()
        
        session = lt.session()
        session.listen_on(6881, 6891)
        
        info = lt.torrent_info(torrent_file_path)
        handle = session.add_torrent({
            'ti': info,
            'save_path': self.download_path
        })
        
        # Clean up temporary torrent file if it was downloaded
        if torrent_file_path != self.torrent_path:
            try:
                os.remove(torrent_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary torrent file: {str(e)}")
        
        logger.info(f"Starting torrent download: {info.name()}")
        start_time = datetime.now()
        last_progress = -1
        last_progress_time = datetime.now()
        
        while not handle.is_seed():
            status = handle.status()
            
            # Check if max retry time exceeded
            if datetime.now() - start_time > timedelta(hours=self.max_retry_hours):
                logger.error("Maximum retry time exceeded")
                return None
            
            # Log progress only when it changes
            current_progress = int(status.progress * 100)
            if current_progress != last_progress:
                logger.info(f'Download progress: {current_progress}%')
                logger.info(f'Download speed: {status.download_rate / 1024:.2f} KB/s')
                logger.info(f'Peers: {status.num_peers}')
                last_progress = current_progress
                last_progress_time = datetime.now()
            
            # If no progress for a long time, log but continue waiting
            if datetime.now() - last_progress_time > timedelta(hours=1):
                logger.warning("No progress in the last hour. Waiting for peers...")
                last_progress_time = datetime.now()
            
            # Sleep before next check
            time.sleep(self.check_interval)
        
        downloaded_path = Path(self.download_path) / info.name()
        logger.info(f"Download completed: {downloaded_path}")
        return str(downloaded_path)

    def upload_to_s3(self, file_path: str) -> bool:
        """
        Upload the file to S3 using multipart upload.
        Returns True if successful, False otherwise.
        """
        logger.info(f"Starting upload to S3: s3://{self.s3_bucket}/{self.s3_key}")
        
        try:
            # Initialize multipart upload
            mpu = self.s3_client.create_multipart_upload(
                Bucket=self.s3_bucket,
                Key=self.s3_key
            )
            
            file_size = os.path.getsize(file_path)
            parts: List[Dict] = []
            uploaded_bytes = 0
            part_number = 1
            
            # Read and upload parts
            with gzip.open(file_path, 'rb') as f:
                while True:
                    data = f.read(self.part_size)
                    if not data:
                        break
                    
                    # Upload part
                    response = self.s3_client.upload_part(
                        Bucket=self.s3_bucket,
                        Key=self.s3_key,
                        PartNumber=part_number,
                        UploadId=mpu['UploadId'],
                        Body=data
                    )
                    
                    parts.append({
                        'PartNumber': part_number,
                        'ETag': response['ETag']
                    })
                    
                    uploaded_bytes += len(data)
                    progress = (uploaded_bytes / file_size) * 100
                    logger.info(f"Upload progress: {progress:.2f}% (Part {part_number})")
                    
                    part_number += 1
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.s3_bucket,
                Key=self.s3_key,
                UploadId=mpu['UploadId'],
                MultipartUpload={'Parts': parts}
            )
            
            logger.info("Upload completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            if 'mpu' in locals():
                logger.info("Aborting multipart upload...")
                self.s3_client.abort_multipart_upload(
                    Bucket=self.s3_bucket,
                    Key=self.s3_key,
                    UploadId=mpu['UploadId']
                )
            return False

    def run(self) -> bool:
        """
        Run the complete service workflow.
        Returns True if successful, False otherwise.
        """
        try:
            # Download torrent
            downloaded_path = self.wait_for_torrent_download()
            if not downloaded_path:
                logger.error("Torrent download failed")
                return False
            
            # Upload to S3
            upload_success = self.upload_to_s3(downloaded_path)
            
            if upload_success:
                logger.info(f"File retained at: {downloaded_path}")
            
            return upload_success
            
        except Exception as e:
            logger.error(f"Service failed: {str(e)}")
            return False

def main():
    # Example usage
    service = TorrentS3Service(
        torrent_path='https://academictorrents.com/download/5c5deeb6cfe1c944044367d2e7465fd8bd2f4acf.torrent',
        download_path='~/downloads',
        s3_bucket='cs229-dota',
        s3_key='downloads/yasp-dump-3_5m.json.gz',
        log_file='~/torrent_service.log',
        max_retry_hours=200,  # Adjust based on your needs
        check_interval=300,  # Check progress every 5 minutes
        part_size_mb=200    # Adjust based on memory constraints
    )
    
    success = service.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
