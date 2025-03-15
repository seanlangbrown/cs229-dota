'''
Citation: Anthropic. (2024). Claude 3.5 Sonnet [Large Language Model]. Retrieved from https://www.anthropic.com
'''

#!/usr/bin/env python3
import ijson
import json
import argparse
import random
import boto3
import sys
import logging
from botocore.exceptions import ClientError
from typing import Dict, Any, List, Set, Optional, Tuple
import time
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaInferrer:
    """Infers JSON schema from large S3 JSON files with sampling support."""
    
    def __init__(
        self,
        bucket: str,
        key: str,
        sample_rate: float = 1.0,
        max_objects: Optional[int] = None,
        max_array_items: int = 100,
        max_schema_depth: int = 20
    ):
        """
        Initialize the schema inferrer.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            sample_rate: Percentage of objects to sample (0.0-1.0)
            max_objects: Maximum number of objects to process
            max_array_items: Maximum number of array items to examine
            max_schema_depth: Maximum nesting depth to process
        """
        self.bucket = bucket
        self.key = key
        self.sample_rate = max(0.0, min(1.0, sample_rate))  # Clamp between 0 and 1
        self.max_objects = max_objects
        self.max_array_items = max_array_items
        self.max_schema_depth = max_schema_depth
        self.s3_client = boto3.client('s3')
        
        # Schema tracking
        self.schema = {"type": "object", "properties": {}}
        self.array_schemas = {}  # Track schemas for arrays
        self.objects_processed = 0
        self.objects_sampled = 0
        self.array_items_sampled = {}  # Track arrays we've sampled
        
        # Statistics
        self.start_time = None
        self.total_objects = 0
        self.bytes_processed = 0
        
    def _merge_type(self, existing_type: str, new_type: str) -> str:
        """Merge two types, using 'mixed' if they differ."""
        if existing_type == new_type:
            return existing_type
        return "mixed"
        
    def _update_schema_field(self, schema_obj: Dict, field: str, value_type: str, 
                            value: Any, path: List[str]) -> None:
        """Update schema with field information."""
        if field not in schema_obj:
            schema_obj[field] = {"type": value_type}
            
            # For objects and arrays, initialize nested structure
            if value_type == "object":
                schema_obj[field]["properties"] = {}
            elif value_type == "array" and value and len(value) > 0:
                # Initialize array item tracking
                array_path = ".".join(path + [field])
                if array_path not in self.array_items_sampled:
                    self.array_items_sampled[array_path] = 0
                schema_obj[field]["items"] = {"type": "unknown"}
        else:
            # Update type if it changes
            existing_type = schema_obj[field]["type"]
            schema_obj[field]["type"] = self._merge_type(existing_type, value_type)
            
    def _process_value(self, schema_obj: Dict, field: str, value: Any, path: List[str]) -> None:
        """Process a JSON value and update the schema."""
        # Detect type
        value_type = "null"
        if value is None:
            value_type = "null"
        elif isinstance(value, bool):
            value_type = "boolean"
        elif isinstance(value, int):
            value_type = "integer"
        elif isinstance(value, float):
            value_type = "number"
        elif isinstance(value, str):
            value_type = "string"
        elif isinstance(value, dict):
            value_type = "object"
        elif isinstance(value, list):
            value_type = "array"
        
        # Update schema for this field
        self._update_schema_field(schema_obj, field, value_type, value, path)
        
        # Process nested structures
        current_path = path + [field]
        if len(current_path) > self.max_schema_depth:
            logger.warning(f"Max schema depth exceeded at {'.'.join(current_path)}")
            return
            
        if value_type == "object" and isinstance(value, dict):
            # Process object fields
            for key, val in value.items():
                self._process_value(schema_obj[field]["properties"], key, val, current_path)
                
        elif value_type == "array" and isinstance(value, list) and value:
            # Process array items (sample if needed)
            array_path = ".".join(current_path)
            
            # Ensure the array path exists in our tracking dictionary
            if array_path not in self.array_items_sampled:
                self.array_items_sampled[array_path] = 0
                
            items_to_sample = min(len(value), self.max_array_items)
            
            # Check if we already sampled enough from this array
            if self.array_items_sampled[array_path] >= self.max_array_items:
                return
                
            # Determine how many more items we can sample
            remaining_samples = self.max_array_items - self.array_items_sampled[array_path]
            items_to_sample = min(items_to_sample, remaining_samples)
            
            # Sample items
            if len(value) > items_to_sample:
                sampled_items = random.sample(value, items_to_sample)
            else:
                sampled_items = value
                
            self.array_items_sampled[array_path] += items_to_sample
            
            # Process each sampled item
            for i, item in enumerate(sampled_items):
                # For the first item, directly update array items schema
                if i == 0 or "items" not in schema_obj[field]:
                    # Determine item type
                    if item is None:
                        item_type = "null"
                    elif isinstance(item, bool):
                        item_type = "boolean"
                    elif isinstance(item, int):
                        item_type = "integer"
                    elif isinstance(item, float):
                        item_type = "number"
                    elif isinstance(item, str):
                        item_type = "string"
                    elif isinstance(item, dict):
                        item_type = "object"
                    elif isinstance(item, list):
                        item_type = "array"
                    
                    # Initialize or update item schema
                    if "items" not in schema_obj[field]:
                        schema_obj[field]["items"] = {"type": item_type} # type: ignore
                        if item_type == "object": # type: ignore
                            schema_obj[field]["items"]["properties"] = {}
                    else:
                        existing_type = schema_obj[field]["items"]["type"]
                        schema_obj[field]["items"]["type"] = self._merge_type(existing_type, item_type) # type: ignore
                
                # Process object items recursively
                if isinstance(item, dict) and schema_obj[field]["items"]["type"] in ["object", "mixed"]:
                    if "properties" not in schema_obj[field]["items"]:
                        schema_obj[field]["items"]["properties"] = {}
                    for k, v in item.items():
                        self._process_value(schema_obj[field]["items"]["properties"], k, v, current_path + ["items"])
    
    def _should_sample_object(self) -> bool:
        """Determine if we should sample the current object based on sample rate."""
        if self.sample_rate >= 1.0:
            return True
        return random.random() <= self.sample_rate
    
    def _save_sample_object(self, obj, filename: str) -> None:
        """Save a sample object to disk."""
        try:
            with open(filename, 'w') as f:
                json.dump(obj, f, indent=2)
            logger.info(f"Saved sample object to {filename}")
        except Exception as e:
            logger.error(f"Error saving sample object: {str(e)}")
    
    def _detect_json_structure(self, stream) -> Tuple[str, Any]:
        """
        Detect the structure of the JSON file (array, object, or ndjson).
        
        Returns:
            Tuple of (structure_type, new_stream)
        """
        try:
            # Read a small chunk to detect structure without using seek
            chunk = stream.read(1024)
            
            # After reading from the stream, we need to make a new request
            # since S3 streams don't support seek operations
            logger.info("Detecting JSON structure from first 1KB")
            
            if not chunk:
                logger.error("Empty file")
                response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
                return "unknown", response['Body']
                
            # Skip whitespace
            i = 0
            while i < len(chunk) and chunk[i:i+1].isspace():
                i += 1
                
            if i >= len(chunk):
                logger.error("File contains only whitespace")
                response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
                return "unknown", response['Body']
                
            first_char = chunk[i:i+1].decode('utf-8')
            
            structure = "unknown"
            if first_char == '[':
                structure = "array"
            elif first_char == '{':
                # Check if it's a single object or newline-delimited
                for j in range(i+1, len(chunk)):
                    if j+1 < len(chunk) and chunk[j:j+1].decode('utf-8') == '\n' and chunk[j+1:j+2].decode('utf-8') == '{':
                        structure = "ndjson"
                        break
                if structure == "unknown":
                    structure = "object"
            else:
                logger.warning(f"Unexpected first character: {first_char}, assuming ndjson")
                structure = "ndjson"
                
            # We've consumed part of the stream, so we need to make a new request
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
            return structure, response['Body']
                
        except Exception as e:
            logger.warning(f"Error detecting JSON structure: {str(e)}, assuming ndjson")
            # Make a new request since we may have consumed part of the stream
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
            return "ndjson", response['Body']
            
    def infer_schema_from_s3(self) -> Dict[str, Any]:
        """
        Stream JSON from S3 and infer its schema with sampling.
        
        Returns:
            Dict representing the inferred JSON schema
        """
        self.start_time = time.time()
        
        try:
            logger.info(f"Starting schema inference on s3://{self.bucket}/{self.key}")
            logger.info(f"Sample rate: {self.sample_rate * 100:.1f}%, Max objects: {self.max_objects or 'unlimited'}")
            
            # Get the object from S3
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
            file_size = response.get('ContentLength', 0)
            
            if file_size:
                logger.info(f"File size: {file_size / (1024 * 1024 * 1024):.2f} GB")
            
            # Stream and parse JSON - get fresh stream after detection
            json_structure, fresh_stream = self._detect_json_structure(response['Body'])
            logger.info(f"Detected JSON structure: {json_structure}")
            
            if json_structure == "array":
                logger.info("Processing JSON array structure")
                self._process_json_array(fresh_stream)
            elif json_structure == "object":
                logger.info("Processing single JSON object")
                self._process_single_object(fresh_stream)
            elif json_structure == "ndjson":
                logger.info("Processing newline-delimited JSON")
                self._process_ndjson(fresh_stream)
            else:
                logger.error(f"Unknown JSON structure: {json_structure}")
                return self.schema
            
            elapsed_time = time.time() - self.start_time
            logger.info(f"Schema inference completed in {elapsed_time:.2f} seconds")
            logger.info(f"Processed {self.objects_processed} objects, sampled {self.objects_sampled} objects")
            logger.info(f"Processed {self.bytes_processed / (1024 * 1024):.2f} MB")
            
            return self.schema
            
        except ClientError as e:
            logger.error(f"Error accessing S3 object: {str(e)}")
            raise
    
    def _process_json_array(self, stream) -> None:
        """Process a JSON file containing an array of objects."""
        # Use ijson to stream array elements
        objects = ijson.items(stream, 'item')
        sample_saved = False
        
        for obj in objects:
            self.objects_processed += 1
            
            # Save the first object as a sample
            if not sample_saved:
                self._save_sample_object(obj, "array_sample.json")
                sample_saved = True
            
            # Apply sampling
            if self._should_sample_object():
                self.objects_sampled += 1
                # Process each field in the object
                if isinstance(obj, dict):
                    for field, value in obj.items():
                        self._process_value(self.schema["properties"], field, value, [])
            
            # Update progress and check limits
            if self.objects_processed % 1000 == 0:
                logger.info(f"Processed {self.objects_processed} objects, sampled {self.objects_sampled}")
                
            if self.max_objects and self.objects_sampled >= self.max_objects:
                logger.info(f"Reached maximum objects limit ({self.max_objects})")
                break
    
    def _process_single_object(self, stream) -> None:
        """Process a JSON file containing a single object."""
        # Parse the object
        obj = json.load(stream)
        self.objects_processed += 1
        self.objects_sampled += 1
        
        # Process each field
        if isinstance(obj, dict):
            for field, value in obj.items():
                self._process_value(self.schema["properties"], field, value, [])
        
        logger.info("Processed single JSON object")
    
    def _process_ndjson(self, stream) -> None:
        """Process a newline-delimited JSON file."""
        # Process each line as a separate JSON object
        line_number = 0
        bytes_read = 0
        sample_saved = False
        
        for line in stream:
            line_number += 1
            bytes_read += len(line)
            self.bytes_processed += len(line)
            
            # Skip empty lines
            if not line.strip():
                continue
                
            try:
                obj = json.loads(line)
                self.objects_processed += 1
                
                # Save the first object as a sample
                if not sample_saved:
                    self._save_sample_object(obj, "ndjson_sample.json")
                    sample_saved = True
                
                # Apply sampling
                if self._should_sample_object():
                    self.objects_sampled += 1
                    # Process each field in the object
                    if isinstance(obj, dict):
                        for field, value in obj.items():
                            self._process_value(self.schema["properties"], field, value, [])
                
                # Update progress and check limits
                if self.objects_processed % 1000 == 0:
                    logger.info(f"Processed {self.objects_processed} objects, sampled {self.objects_sampled}")
                    logger.info(f"Processed {self.bytes_processed / (1024 * 1024):.2f} MB")
                    
                if self.max_objects and self.objects_sampled >= self.max_objects:
                    logger.info(f"Reached maximum objects limit ({self.max_objects})")
                    break
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing JSON on line {line_number}: {str(e)}")
    
    def save_schema(self, output_file: str) -> None:
        """Save the inferred schema to a file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.schema, f, indent=2)
            logger.info(f"Schema saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving schema: {str(e)}")

def main():
    # Hardcoded parameters
    bucket = 'cs229-dota'
    key = 'data/yasp-dump-90gb.json'
    output_file = './schema_yasp-dump-90gb.json'
    
    # Sampling parameters
    sample_rate = 0.1  # Sample 10% of objects
    max_objects = 10000  # Process up to 10,000 objects
    max_array_items = 100  # Sample up to 100 items per array
    max_schema_depth = 20  # Maximum nesting level
    
    # Create parser for optional overrides
    parser = argparse.ArgumentParser(description='Infer JSON schema from large S3 files with sampling')
    parser.add_argument('--sample-rate', type=float, help='Override default sample rate (0.0-1.0)')
    parser.add_argument('--max-objects', type=int, help='Override maximum number of objects to process')
    
    args = parser.parse_args()
    
    # Override defaults if specified
    if args.sample_rate is not None:
        sample_rate = args.sample_rate
    if args.max_objects is not None:
        max_objects = args.max_objects
    
    # Log the parameters being used
    logger.info(f"Using bucket: {bucket}")
    logger.info(f"Using key: {key}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Sample rate: {sample_rate}")
    logger.info(f"Max objects: {max_objects}")
    
    inferrer = SchemaInferrer(
        bucket=bucket,
        key=key,
        sample_rate=sample_rate,
        max_objects=max_objects,
        max_array_items=max_array_items,
        max_schema_depth=max_schema_depth
    )
    
    schema = inferrer.infer_schema_from_s3()
    inferrer.save_schema(output_file)

if __name__ == "__main__":
    main()