'''
Citation: Anthropic. (2024). Claude 3.5 Sonnet [Large Language Model]. Retrieved from https://www.anthropic.com
'''

#!/usr/bin/env python3
import boto3
import json
import argparse
import sys
import logging
from botocore.exceptions import ClientError

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_file):
    """Load upload information from checkpoint file."""
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return data.get('upload_id'), data.get('last_part_number', 0)
    except Exception as e:
        logger.error(f"Failed to load checkpoint file: {str(e)}")
        return None, 0

def list_active_uploads(s3_client, bucket, key=None):
    """List all active multipart uploads for a bucket/key."""
    try:
        if key:
            logger.info(f"Listing active uploads for s3://{bucket}/{key}")
            response = s3_client.list_multipart_uploads(
                Bucket=bucket,
                Prefix=key
            )
        else:
            logger.info(f"Listing all active uploads for bucket {bucket}")
            response = s3_client.list_multipart_uploads(
                Bucket=bucket
            )
        
        uploads = []
        if 'Uploads' in response:
            for upload in response['Uploads']:
                uploads.append({
                    'UploadId': upload['UploadId'],
                    'Key': upload['Key'],
                    'Initiated': upload['Initiated']
                })
            
            if uploads:
                logger.info(f"Found {len(uploads)} active multipart uploads")
            else:
                logger.info("No active multipart uploads found")
                
        return uploads
    except ClientError as e:
        logger.error(f"Error listing multipart uploads: {str(e)}")
        return []

def abort_upload(s3_client, bucket, key, upload_id):
    """Abort a specific multipart upload."""
    try:
        logger.info(f"Aborting upload {upload_id} for s3://{bucket}/{key}")
        s3_client.abort_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id
        )
        logger.info("Upload aborted successfully")
        return True
    except ClientError as e:
        logger.error(f"Error aborting upload: {str(e)}")
        return False

def delete_checkpoint(checkpoint_file):
    """Delete the checkpoint file."""
    try:
        import os
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info(f"Deleted checkpoint file: {checkpoint_file}")
            return True
        else:
            logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            return False
    except Exception as e:
        logger.error(f"Error deleting checkpoint file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Abort S3 multipart uploads from checkpoint or bucket/key')
    
    # Define mutually exclusive group for operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--checkpoint', action='store_true', help='Abort upload using checkpoint file')
    mode_group.add_argument('--list', action='store_true', help='List all active uploads for the bucket')
    mode_group.add_argument('--abort', action='store_true', help='Abort upload with specified upload-id')
    
    # Optional arguments
    parser.add_argument('--upload-id', help='Upload ID to abort (required with --abort)')
    parser.add_argument('--delete-checkpoint', action='store_true', 
                        help='Delete checkpoint file after aborting upload')
    
    args = parser.parse_args()
    
    # Hardcoded parameters
    bucket = 'cs229-dota'
    key = 'data/yasp-dump-90gb.json'
    checkpoint_file = './upload_checkpoint.json'
    
    # Initialize boto3 client
    s3_client = boto3.client('s3')
    
    # Mode 1: Abort upload from checkpoint
    if args.checkpoint:
        # Load upload_id from checkpoint
        upload_id, last_part_number = load_checkpoint(checkpoint_file)
        
        if not upload_id:
            logger.error("No valid upload_id found in checkpoint file")
            return 1
            
        # Abort the upload
        success = abort_upload(s3_client, bucket, key, upload_id)
        
        # Delete checkpoint if requested
        if success and args.delete_checkpoint:
            delete_checkpoint(checkpoint_file)
            
        return 0 if success else 1
    
    # Mode 2: List active uploads
    elif args.list:
        # List active uploads for the bucket
        uploads = list_active_uploads(s3_client, bucket, key)
        
        if uploads:
            print("\nActive Multipart Uploads:")
            print("------------------------")
            for i, upload in enumerate(uploads, 1):
                print(f"{i}. Key: {upload['Key']}")
                print(f"   Upload ID: {upload['UploadId']}")
                print(f"   Initiated: {upload['Initiated']}")
                print()
        else:
            print("No active multipart uploads found.")
            
        return 0
    
    # Mode 3: Abort specific upload by ID
    elif args.abort:
        if not args.upload_id:
            logger.error("Upload ID (--upload-id) is required with --abort")
            return 1
            
        # Abort the upload
        success = abort_upload(s3_client, bucket, key, args.upload_id)
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())