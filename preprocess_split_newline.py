#!/usr/bin/env python3
import gzip
import ijson
import json
import os
import time
import boto3
from botocore.exceptions import ClientError
import logging
from datetime import datetime

from decimal import Decimal
import numpy as np

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('json_processor.log'),
        logging.StreamHandler()  # This will output to console too
    ]
)
logger = logging.getLogger('json_processor')

# Print confirmation that logging is set up
print("Logs will print to json_processor.log")


# Use float32 everywhere for pandas performance later
class DecimalOrNumberFloat32Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return np.float32(float(o)).item()
        if isinstance(o, (int, float)):
            return np.float32(o).item()
        return super().default(o)
    

def setup_s3_client():
    """Set up and return an S3 client."""
    try:
        client = boto3.client('s3')
        print("S3 client initialized successfully")
        return client
    except Exception as e:
        print(f"Error initializing S3 client: {str(e)}")
        raise

def save_checkpoint(processed_count, match_id_start_times, chat_events, match_id_has_tower_kill, lane_pos_stats=None, checkpoint_file='checkpoint.json'):
    """Save a checkpoint of processed objects count and match_id data."""
    checkpoint = {
        'processed_count': processed_count,
        'match_id_start_times': match_id_start_times,
        'match_id_has_tower_kill': match_id_has_tower_kill,
        'chat_events': list(chat_events),
    }
    
    # Add lane position stats if provided
    if lane_pos_stats:
        checkpoint['lane_pos_stats'] = lane_pos_stats
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, cls=DecimalOrNumberFloat32Encoder)
    
    logger.info(f"Checkpoint saved: {processed_count} objects processed")

def load_checkpoint(checkpoint_file='checkpoint.json'):
    """Load a checkpoint if it exists.

    Returns:
    (processed_count, match_id_start_times, match_id_has_tower_kill, chat_events, lane_pos_stats)
    """
    default_lane_pos_stats = {'total_players': 0, 'players_with_lane_pos': 0}
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint['processed_count']} objects previously processed")
            
            # Extract lane position stats if present
            lane_pos_stats = checkpoint.get('lane_pos_stats', default_lane_pos_stats)
            
            return checkpoint['processed_count'], checkpoint['match_id_start_times'], checkpoint['match_id_has_tower_kill'], set(checkpoint['chat_events']), lane_pos_stats
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
    
    return 0, {}, {}, set(), default_lane_pos_stats

def upload_to_s3(s3_client, data, bucket, key, retry_count=0, max_retries=3):
    """Upload data to S3 with retry logic."""
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=data)
        return True
    except ClientError as e:
        if retry_count < max_retries:
            # Exponential backoff
            wait_time = 2 ** retry_count
            logger.warning(f"S3 upload failed, retrying in {wait_time} seconds. Error: {str(e)}")
            time.sleep(wait_time)
            return upload_to_s3(s3_client, data, bucket, key, retry_count + 1, max_retries)
        else:
            logger.error(f"Failed to upload to S3 after {max_retries} retries. Bucket: {bucket}, Key: {key}, Error: {str(e)}")
            return False

def update_match_object(current_object, current_match_id, match_id_has_tower_kill, chat_events):
    """
    Update the match object with tower kill information and process player data.
    
    Args:
        current_object: The current match object being processed
        current_match_id: ID of the current match

        match_id_has_tower_kill: Dictionary tracking which match IDs have tower kills
        chat_events: Set of all chat events found so far
        
    Returns:
        tuple: (has_tower_kill, total_players_in_object, players_with_lane_pos_in_object)
    """
    # Safety check for None values
    if current_object is None or current_match_id is None:
        return False, 0, 0
        
    # Track player counts for this object
    total_players_in_object = 0
    players_with_lane_pos_in_object = 0
    has_tower_kill = False
    
    # Check if any of the chat items are tower kill events

    tower_kill_events = []
    barracks_kill_events = []
    roshan_kill_events = []

    # "objectives": [{"time": 1971, "type": "chat_event", "subtype": "CHAT_MESSAGE_TOWER_KILL", "value": 3, "player1": 8, "player2": -1, "team": 3, "slot": 8},
    if 'objectives' in current_object and current_object['objectives'] != None:
        for obj_item in current_object['objectives']:
            if obj_item.get('subtype') == None:
                continue
            
            if len(chat_events) < 1001:
                if obj_item.get('subtype').startswith('CHAT_MESSAGE_'):
                    chat_events.add(obj_item.get('subtype'))
                
            if obj_item.get('subtype') == 'CHAT_MESSAGE_TOWER_KILL':
                has_tower_kill = True
                # Add to tower_kill_events for player processing
                tower_kill_events.append({
                    'slot': obj_item.get('slot'),
                    'time': obj_item.get('time'),
                    'value': obj_item.get('value'),
                    'team': obj_item.get('team'),
                    'player1': obj_item.get('player1'),
                    'player2': obj_item.get('player2'),
                })

        # # "subtype": "CHAT_MESSAGE_BARRACKS_KILL", "value": 512, "player1": -1, "player2": -1, "key": "512"
        # if obj_item.get('subtype') == 'CHAT_MESSAGE_BARRACKS_KILL':
        #     # Add to barracks_kill_events for player processing
        #     # NOTE appears to be incomplete with no team or other information?
        #     # barracks_kill_events.append({
        #     #     'time': obj_item.get('time'),
        #     #     'value': obj_item.get('value'),
        #     #     'team': obj_item.get('team'),
        #     #     'player1': obj_item.get('player1'),
        #     #     'player2': obj_item.get('player2'),
        #     # })
        #     pass

            # "CHAT_MESSAGE_ROSHAN_KILL", "value": 200, "player1": 3, "player2": -1, "team": 3
            if obj_item.get('subtype') == 'CHAT_MESSAGE_ROSHAN_KILL':
                # Add to roshan_kill_events for player processing
                roshan_kill_events.append({
                    'time': obj_item.get('time'),
                    'value': obj_item.get('value'),
                    'team': obj_item.get('team'),
                    'player1': obj_item.get('player1'),
                    'player2': obj_item.get('player2'),
                })
    
    # Add tower_killed flag
    current_object['tower_killed'] = has_tower_kill

    has_lane_pos = False
    
    # Check for lane_pos data in players
    if 'players' in current_object:
        for player in current_object['players']:
            total_players_in_object += 1
            
            # Check if lane_pos has at least one non-null property
            if ('lane_pos' in player and 
                isinstance(player['lane_pos'], dict) and 
                any(player['lane_pos'].values())):
                players_with_lane_pos_in_object += 1
                has_lane_pos = True
    
    has_unattributed_tower_kill = False

    # Update match_id_has_tower_kill if this object has tower kill - only set once
    if has_tower_kill:

        current_object['tower_kills'] = []

        for event_data in tower_kill_events:
            slot = event_data.get('slot')
            time = event_data.get('time')
            value = event_data.get('value')
            player1 = event_data.get('player1')
            player2 = event_data.get('player2')
            team = event_data.get('team')

            current_object['tower_kills'].append({
                            'time': time,
                            'value': value,
                            'player1': player1,
                            'slot': slot,
                            'player2': player2,
                            'team': team,
                        })
        
            # Process tower kills and update player data
            if 'players' in current_object:
                # Process collected tower kill events for players
           
                
                # Find matching player by slot (regardless of time value)
                if slot is not None:
                    # Find matching player by slot
                    player_found = False
                    for player in current_object['players']:
                        if player.get('player_slot') == slot or player.get('player_slot') == slot + 123: # slots are numbered starting from 128 for players 5-9
                            # Add tower_kill flag to player
                            player['tower_kill'] = True
                            
                            # Initialize tower_kills array if it doesn't exist
                            if 'tower_kills' not in player:
                                player['tower_kills'] = []
                            
                            # Append time to tower_kills array (might be None)
                            player['tower_kills'].append({
                                'time': time,
                                'value': value,
                                # 'player1': player1, # equal to slot (current player)
                                'player2': player2,
                                'team': team,
                            })

                            
                            
                            # Player found, no need to continue searching
                            player_found = True
                            break
                            
                    # Log if player not found
                    if not player_found:
                        logger.warning(f"Player with slot {slot} not found in match for tower kill {current_match_id}")
                        has_unattributed_tower_kill = True
    
    # Process roshan kills and update player data
    current_object['roshan_kills'] = []
    
    for event_data in roshan_kill_events:
        time = event_data.get('time')
        value = event_data.get('value')
        team = event_data.get('team')
        player1 = event_data.get('player1')
        player2 = event_data.get('player2')

        current_object['roshan_kills'].append({
                            'time': time,
                            'value': value,
                            'player1': player1,
                            'player2': player2,
                            'team': team,
                        })
        

        if 'players' in current_object:
            # Process collected tower kill events for players
        
            # Find matching player by slot (regardless of time value)
            if player1 is not None:
                # Find matching player by slot
                player_found = False
                for player in current_object['players']:
                    if player.get('player_slot') == player1 or player.get('player_slot') == player1 + 123:
                        # Add tower_kill flag to player
                        player['roshan_kill'] = True
                        
                        # Initialize tower_kills array if it doesn't exist
                        if 'roshan_kills' not in player:
                            player['roshan_kills'] = []
                        
                        # Append time to tower_kills array (might be None)
                        player['roshan_kills'].append({
                            'time': time,
                            'value': value,
                            # 'player1': player1, # 3 current player
                            'player2': player2,
                        })
                        
                        # Player found, no need to continue searching
                        player_found = True
                        break
                        
                # Log if player not found
                if not player_found:
                    logger.warning(f"Player with slot {player1} not found in match for roshan kill {current_match_id}")

    has_objectives = current_object['objectives'] != None and len(current_object['objectives']) > 0
    has_teamfights = current_object['teamfights'] != None and len(current_object['teamfights']) > 0


    match_id_has_tower_kill[current_match_id] = {
        'tower_kill': has_tower_kill,
        'version': current_object['version'],
        'has_objectives': has_objectives,
        'has_teamfights': has_teamfights,
        'has_lan_pos': has_lane_pos,
        'has_unattributed_tower_kills': has_unattributed_tower_kill
    }
    
    return has_tower_kill, total_players_in_object, players_with_lane_pos_in_object

def process_json_file(input_file, s3_bucket, checkpoint_interval=100, local_output_dir=None, local_match_limit=0, exit_after_local=False):
    """
    Process a large gzipped JSON file and upload processed data to S3.
    
    Args:
        input_file: Path to the gzipped JSON file
        s3_bucket: S3 bucket for uploading results
        checkpoint_interval: How often to save checkpoints (number of objects)
        local_output_dir: Directory to store local copies of JSON files
        local_match_limit: Number of match_ids to save locally (0 for none)
        exit_after_local: If True, exit after local_match_limit is reached
    """
    # Get script directory and resolve paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve input file path relative to script location
    abs_input_file = os.path.abspath(os.path.join(script_dir, input_file))
    
    # Check if file exists
    if not os.path.exists(abs_input_file):
        logger.error(f"Input file not found at {abs_input_file}")
        raise FileNotFoundError(f"Input file not found at {abs_input_file}")
    else:
        print(f"Found input file: {abs_input_file}")

    # Resolve local output directory if provided
    if local_output_dir:
        abs_local_output_dir = os.path.abspath(os.path.join(script_dir, local_output_dir))
        print(f"Output directory will be: {abs_local_output_dir}")
    else:
        abs_local_output_dir = None
    
    s3_client = setup_s3_client()
    
    # Load checkpoint if it exists
    processed_count, match_id_start_times, match_id_has_tower_kill, chat_events, lane_pos_stats = load_checkpoint()
    
    # Statistics
    total_objects = processed_count
    tower_kill_count = 0
    skipped_missing_match_id = 0
    skipped_missing_match_seq_num = 0
    
    # Lane position tracking
    total_players = lane_pos_stats['total_players']
    players_with_lane_pos = lane_pos_stats['players_with_lane_pos']
    
    # Track match IDs to save locally
    match_ids_to_save_locally = set()
    should_exit_after_object = False
    
    # Create local output directory if needed
    if abs_local_output_dir and local_match_limit > 0:
        os.makedirs(abs_local_output_dir, exist_ok=True)
        # Create subdirectories
        os.makedirs(os.path.join(abs_local_output_dir, "preprocess_stats"), exist_ok=True)
        os.makedirs(os.path.join(abs_local_output_dir, "preprocess/yes_tower_kills"), exist_ok=True)
        os.makedirs(os.path.join(abs_local_output_dir, "preprocess/no_tower_kills"), exist_ok=True)
    
    try:
        # Use context manager to ensure file is properly closed
        with gzip.open(abs_input_file, 'rb') as f:
            # Stream through objects one at a time using ijson's items function
            # This provides complete parsed objects without manual path building
            objects = ijson.items(f, 'item')
            
            for current_object in objects:
                # Extract required fields from the complete object
                current_match_id = current_object.get('match_id')
                current_match_seq_num = current_object.get('match_seq_num')
                
                # Skip processing if match_id or match_seq_num is missing
                if current_match_id is None:
                    logger.warning("Skipping object with missing match_id")
                    skipped_missing_match_id += 1
                    continue
                    
                if current_match_seq_num is None:
                    logger.warning("Skipping object with missing match_seq_num")
                    skipped_missing_match_seq_num += 1
                    continue
                
                # Add match_id to tower kill tracking if not already there
                # if current_match_id not in match_id_has_tower_kill:
                #     match_id_has_tower_kill[current_match_id] = False
                
                # Track for local saving
                if local_match_limit > 0 and len(match_ids_to_save_locally) < local_match_limit:
                    if current_match_id not in match_ids_to_save_locally:
                        match_ids_to_save_locally.add(current_match_id)
                
                # Check start_time for min calculation
                start_time = current_object.get('start_time')
                if start_time is not None:
                    if current_match_id not in match_id_start_times:
                        match_id_start_times[current_match_id] = start_time
                    else:
                        # Keep the minimum start_time
                        if start_time < match_id_start_times[current_match_id]:
                            match_id_start_times[current_match_id] = start_time
                
                # Generate a unique ID for this object
                object_id = f"{current_match_id}_{current_match_seq_num}"
                
                # Update the match object with tower kill and player data
                has_tower_kill, object_players, object_players_with_lane_pos = update_match_object(
                    current_object, 
                    current_match_id, 
                    match_id_has_tower_kill,
                    chat_events
                )
                
                # Update player statistics
                total_players += object_players
                players_with_lane_pos += object_players_with_lane_pos
                
                # Determine the appropriate S3 path
                if has_tower_kill:
                    tower_kill_count += 1
                    s3_key = f"preprocess/yes_tower_kills/{current_match_id}/{current_match_id}_{current_match_seq_num}.json"
                else:
                    s3_key = f"preprocess/no_tower_kills/{current_match_id}/{current_match_id}_{current_match_seq_num}.json"
                
                # Prepare JSON data once to avoid multiple serializations
                json_data = json.dumps(current_object, cls=DecimalOrNumberFloat32Encoder)
                
                # Upload to S3
                success = upload_to_s3(
                    s3_client, 
                    json_data, 
                    s3_bucket, 
                    s3_key
                )
                
                if not success:
                    error_key = f"errors/{current_match_id}_{current_match_seq_num}_error.json"
                    error_data = {
                        'error': 'Failed to upload after max retries',
                        'match_id': current_match_id,
                        'match_seq_num': current_match_seq_num,
                        'object': current_object
                    }
                    upload_to_s3(s3_client, json.dumps(error_data, cls=DecimalOrNumberFloat32Encoder), s3_bucket, error_key)
                    logger.error(f"Upload failed for {object_id}, error logged to S3")
                    raise Exception(f"Failed to upload {object_id} after maximum retries")
                
                # Save locally if this match_id is in our list to save
                if abs_local_output_dir and current_match_id in match_ids_to_save_locally:
                    local_path = os.path.join(abs_local_output_dir, s3_key)
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    # Write the file
                    try:
                        with open(local_path, 'w') as f:
                            f.write(json_data)
                    except Exception as e:
                        logger.error(f"Error writing local file {local_path}: {str(e)}")
                        
                    # Check if we've reached the limit and should exit
                    if exit_after_local and len(match_ids_to_save_locally) >= local_match_limit:
                        logger.info(f"Reached local_match_limit of {local_match_limit}. Preparing to exit...")
                        # Flag that we should prepare to exit soon
                        should_exit_after_object = True
                
                # Increment processed count
                total_objects += 1
                
                # Save checkpoint periodically
                if total_objects % checkpoint_interval == 0:
                    lane_pos_stats = {
                        'total_players': total_players,
                        'players_with_lane_pos': players_with_lane_pos
                    }
                    save_checkpoint(total_objects, match_id_start_times, chat_events, match_id_has_tower_kill, lane_pos_stats)
                    logger.info(f"Processed {total_objects} objects, {tower_kill_count} with tower kills, {len(match_id_has_tower_kill)} unique match IDs")
                
                # Check if we should exit after processing this object
                if should_exit_after_object:
                    logger.info(f"Reached local limit, stopping")
            
                    break
            
            # After processing all objects, save final results
            logger.info(f"Processing complete: {total_objects} total objects, {tower_kill_count} with tower kills")
            logger.info(f"Found {len(match_id_has_tower_kill)} unique match IDs")
            
            # Upload match metadata 
            tower_kill_json = json.dumps(match_id_has_tower_kill, cls=DecimalOrNumberFloat32Encoder)
            upload_to_s3(
                s3_client,
                tower_kill_json,
                s3_bucket,
                "preprocess_stats/match_ids_with_metadata.json"
            )
            
            # Save locally if configured
            if abs_local_output_dir and local_match_limit > 0:
                local_path = os.path.join(abs_local_output_dir, "preprocess_stats/match_ids_with_metadata.json")
                try:
                    with open(local_path, 'w') as f:
                        f.write(tower_kill_json)
                except Exception as e:
                    logger.error(f"Error writing local file {local_path}: {str(e)}")
            
            # Create and upload the array of match_id and start_time pairs
            start_time_array = [
                {"match_id": match_id, "start_time": start_time, "tower_kills": match_id_has_tower_kill.get(match_id, False)}
                for match_id, start_time in match_id_start_times.items()
            ]
            # Sort by start_time
            start_time_array.sort(key=lambda x: x["start_time"])
            start_times_json = json.dumps(start_time_array, cls=DecimalOrNumberFloat32Encoder)
            
            
            
            # Save locally if configured
            if abs_local_output_dir and local_match_limit > 0:
                local_path = os.path.join(abs_local_output_dir, "preprocess_stats/match_ids_with_start_times.json")
                try:
                    with open(local_path, 'w') as f:
                        f.write(start_times_json)
                except Exception as e:
                    logger.error(f"Error writing local file {local_path}: {str(e)}")
            
            # Create and upload processing statistics
            percent_players_with_lane_pos = 0
            if total_players > 0:
                percent_players_with_lane_pos = (players_with_lane_pos / total_players) * 100
                
            stats = {
                "total_objects_processed": total_objects,
                "objects_with_tower_kill_events": tower_kill_count,
                "unique_match_ids": len(match_id_has_tower_kill),
                "total_players": total_players,
                "percent_players_with_lane_pos": percent_players_with_lane_pos,
                "skipped_missing_match_id": skipped_missing_match_id,
                "skipped_missing_match_seq_num": skipped_missing_match_seq_num,
                "processed_timestamp": datetime.now().isoformat(),
                "chat_events": list(chat_events)
            }
            stats_json = json.dumps(stats, cls=DecimalOrNumberFloat32Encoder)
            
            
            
            # Save locally if configured
            if abs_local_output_dir:
                local_path = os.path.join(abs_local_output_dir, "preprocess_stats/processing_statistics.json")
                try:
                    with open(local_path, 'w') as f:
                        f.write(stats_json)
                except Exception as e:
                    logger.error(f"Error writing local file {local_path}: {str(e)}")
                    
            # Log information about locally saved files
            if abs_local_output_dir:
                logger.info(f"Saved data for {len(match_ids_to_save_locally)} (all) match IDs locally to {abs_local_output_dir}")
                print(f"Saved data for {len(match_ids_to_save_locally)} (all) match IDs locally to {abs_local_output_dir}")


            # upload to s3
            upload_to_s3(
                s3_client,
                start_times_json,
                s3_bucket,
                "preprocess_stats/match_ids_with_start_times.json"
            )

            upload_to_s3(
                s3_client,
                stats_json,
                s3_bucket,
                "preprocess_stats/processing_statistics.json"
            )
            
            # Print final statistics
            print(f"Total objects processed: {total_objects}")
            print(f"Objects with tower kill events: {tower_kill_count}")
            print(f"Unique match IDs: {len(match_id_has_tower_kill)}")
            print(f"Percent of players with lane position data: {percent_players_with_lane_pos:.2f}%")
            print(f"Objects skipped due to missing match_id: {skipped_missing_match_id}")
            print(f"Objects skipped due to missing match_seq_num: {skipped_missing_match_seq_num}")
            print(f"Count of chat events: {len(chat_events)}")
            
            # Clean up checkpoint as processing completed successfully
            if os.path.exists('checkpoint.json'):
                os.remove('checkpoint.json')
                logger.info("Checkpoint file removed as processing completed successfully")
            
    except Exception as e:
        import traceback
        # Get the full stack trace
        stack_trace = traceback.format_exc()
        
        logger.error(f"Error processing file: {str(e)}")
        logger.error(f"Stack trace: {stack_trace}")
        print(f"ERROR: {str(e)}")
        print(f"STACK TRACE:\n{stack_trace}")
        
        # Save checkpoint to resume later
        lane_pos_stats = {
            'total_players': total_players,
            'players_with_lane_pos': players_with_lane_pos
        }
        save_checkpoint(total_objects, match_id_start_times, chat_events, match_id_has_tower_kill, lane_pos_stats)
        
        # Only try to save error details if we have match_id and match_seq_num
        if 'current_match_id' in locals() and 'current_match_seq_num' in locals() and current_object is not None: # type: ignore
            error_log = {
                "error": str(e),
                "match_id": current_match_id, # type: ignore
                "match_seq_num": current_match_seq_num, # type: ignore
                "object": current_object # type: ignore
            }
            with open(f"error_{current_match_id}_{current_match_seq_num}.json", 'w') as f: # type: ignore
                json.dump(error_log, f, cls=DecimalOrNumberFloat32Encoder)
            logger.error(f"Error details saved to error_{current_match_id}_{current_match_seq_num}.json") # type: ignore
        raise

def main():
    # Hardcoded parameters
    INPUT_FILE = "../downloads/yasp-dump.json.gz"
    S3_BUCKET = "cs229-dota"
    CHECKPOINT_INTERVAL = 1000  # Save checkpoint every 1000 objects
    LOCAL_OUTPUT_DIR = "../data/preprocess"  # Directory to store local copies
    LOCAL_MATCH_LIMIT = 200 # Save the first 5 match_ids locally
    EXIT_AFTER_LOCAL = False  # Exit after reaching local_match_limit
    
    print(f"Parameters: INPUT_FILE={INPUT_FILE}, LOCAL_OUTPUT_DIR={LOCAL_OUTPUT_DIR}")
    
    # Process the file
    process_json_file(
        INPUT_FILE, 
        S3_BUCKET, 
        CHECKPOINT_INTERVAL,
        LOCAL_OUTPUT_DIR,
        LOCAL_MATCH_LIMIT,
        EXIT_AFTER_LOCAL
    )

if __name__ == "__main__":
    start_time = datetime.now()
    try:
        main()
        end_time = datetime.now()
        print(f"Processing completed")
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        
        end_time = datetime.now()
        print(f"ERROR: Processing failed after {end_time - start_time}: {str(e)}")
        print(f"STACK TRACE:\n{stack_trace}")
        logger.error(f"Processing failed after {end_time - start_time}: {str(e)}")
        logger.error(f"Stack trace: {stack_trace}")
        
        exit(1)