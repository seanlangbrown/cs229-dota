import boto3
import random
from normalize import RangeTracker, save_range_tracker
import json
from data_loader import transform_json_to_ts, load_file_from_s3
import pandas as pd
from collections import defaultdict
import time
from json_stream_writer import JSONStreamWriter



SAMPLE_MIN_TIME_CUTOFF = 1265932800


def list_all_match_files(min_time=SAMPLE_MIN_TIME_CUTOFF):
    bucket = 'cs229-dota'
    s3_client = boto3.client('s3')

    with JSONStreamWriter('all_match_files_yes.json') as f:
        for prefix in ['preprocess/yes_tower_kills/']: # TODO verify paths, get bucket, 'preprocess/no_tower_kills/'
            paginator = s3_client.get_paginator('list_objects_v2')
            count = 0
            break_out = False
            for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if break_out:
                    break

                for obj in result.get('Contents', []):
                    if obj['Key'].endswith('.json'):
                        timestamp = 'unknown'
                        try:
                            timestamp = obj['Key'].split('/')[-1].split('_')[1].split('.')[0]
                            match_id = obj['Key'].split('/')[-1].split('_')[0]
                            if int(timestamp) <= min_time:
                                continue
                        except Exception:
                            print(f"unknown timestamp {timestamp}")
                            continue

                        if timestamp == 'unknown':
                            continue

                        data = {
                            "time": timestamp,
                            "key": obj['Key']
                        }
                        
                        f.write(match_id, data)
                           
                        count += 1
            print(count)
    
        return


def list_specific_s3_paths_by_tower_kills(sample_rate=None, limit=None, offset=0, min_time=SAMPLE_MIN_TIME_CUTOFF):
    bucket = 'cs229-dota'
    s3_client = boto3.client('s3')

    with_tower_kills = []
    no_tower_kills = []

    for prefix in ['preprocess/yes_tower_kills/']: # TODO verify paths, get bucket, 'preprocess/no_tower_kills/'
        paginator = s3_client.get_paginator('list_objects_v2')
        count = 0
        break_out = False
        offset_i = 0
        for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if break_out:
                break

            for obj in result.get('Contents', []):
                if (sample_rate and random.random() < sample_rate) or offset_i < offset:
                    offset_i += 1
                    continue
                if obj['Key'].endswith('.json'):
                    try:
                        timestamp = obj['Key'].split('/')[-1].split('_')[1].split('.')[0]   
                        if int(timestamp) <= min_time:
                            continue
                    except Exception:
                        print(f"unknown timestamp {timestamp}") # type: ignore
                        continue

                    if 'yes_tower_kills' in obj['Key']:
                        with_tower_kills.append(obj['Key'])
                    else:
                        no_tower_kills.append(obj['Key'])
                    count += 1
                if limit is not None and count >= limit:
                    break_out = True
                    break
   
    return with_tower_kills, no_tower_kills


def count_specific_s3_paths(i=None, limit=None, offset=0):
    bucket = 'cs229-dota'
    s3_client = boto3.client('s3')

    with_tower_kills = defaultdict(int)
    no_tower_kills = defaultdict(int)
    last_key_processed = ""

    prefixes = ['preprocess/yes_tower_kills/', 'preprocess/no_tower_kills/']

    for j in range(len(prefixes)): # TODO verify paths, get bucket
        if i is not None and int(i) != j:
            continue

        prefix = prefixes[j]
        paginator = s3_client.get_paginator('list_objects_v2')
        count = 0
        break_out = False
        offset_i = 0
        for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if break_out:
                break

            for obj in result.get('Contents', []):
                if offset_i < offset:
                    offset_i += 1
                    continue
                if obj['Key'].endswith('.json'):
                    timestamp = None
                    try:
                        timestamp = obj['Key'].split('/')[-1].split('_')[1].split('.')[0]

                        # print(timestamp)

                        day_key = str(pd.to_datetime(timestamp, unit='s').floor('D'))

                        # print(day_key)
                    except Exception:
                        day_key = 'unknown'
                        print(f"unknown timestamp {timestamp}")
                
                    if 'yes_tower_kills' in obj['Key']:
                        with_tower_kills[day_key] += 1
                    else:
                        no_tower_kills[day_key] += 1
                    count += 1

                    last_key_processed = obj['Key']
                # if count is not None and count >= limit:
                #     break_out = True
                #     break

                    if count % 100000 == 0:
                        print(f"Processed {count} items")
                        print(f"Unique days (with tower kills): {len(with_tower_kills)}")
                        print(f"Unique days (no tower kills): {len(no_tower_kills)}")


            # save so far
            with open(f"./count_matches_by_day_{i}.json", 'w') as f:
                f.write(json.dumps({
                    'yes_tower_kills': with_tower_kills,
                    'no_tower_kills': no_tower_kills,
                    'last_key_processed': last_key_processed
                }))
            
            
   
    return with_tower_kills, no_tower_kills


class SampleSelector:
   def __init__(self, full_list):
       self.full_list = full_list.copy()
       self.sample = []
       self.used = set()

   def initial_sample(self, n):
       sample_size = min(n, len(self.full_list))
       candidates = [x for x in self.full_list if x not in self.used]
       self.sample = random.sample(candidates, sample_size)
       self.used.update(self.sample)
       return self.sample

   def add_sample(self):
       candidates = [x for x in self.full_list if x not in self.used]
       if not candidates:
           return None
       new_item = random.choice(candidates)
       self.sample.append(new_item)
       self.used.add(new_item)
       return new_item


def get_normalization_parameters(n):
    ranges = RangeTracker(None)

    missing_data_reports = {}
    report_i = 0

    for i in range(5):
        # 5 tries to get a full sample
        if ranges.get_total_count() >= n:
            continue

        print(f"attempt {i}")

        limit = 100*n + 300
        yes_kills, no_kills = list_specific_s3_paths_by_tower_kills(sample_rate=0.0015, limit=limit, offset=limit * i)

        print(len(yes_kills))
        print(len(no_kills))

        yes_sampler = SampleSelector(yes_kills)
        no_sampler = SampleSelector(no_kills)

        for sampler in [yes_sampler]:
            sample = sampler.initial_sample(n)
            # print(f"sample: {sample}")
            break_out = False
            for path in sample:
                if break_out:
                    break

                curr_path = path
                while True:
                    # loop until a valid sample is found
                    # print(curr_path)
                    try:
                        # load data from s3
                        data = json.loads(load_file_from_s3(path))

                        df, missing_data = transform_json_to_ts(data)

                        # print(missing_data)
                    
                        if (not 'any' in missing_data or not missing_data['any']) and (not 'error' in missing_data or not missing_data['error']):
                            ranges.update(df)
                            break
                        else:
                            missing_data_reports[curr_path] = missing_data

                        if len(missing_data) == 3:
                            if all(key in ['any', 'radiant_xp_adv', 'radiant_gold_adv'] for key in missing_data):
                                # we'll allow it for this sample
                                ranges.update(df)
                                break

                        if len(missing_data_reports) >= 1000:
                            # flush to keep memory usage constant
                            print('flushing missing data')
                            with open(f"./missing_data_found_normalize_{report_i}.json", 'w') as f:
                                f.write(json.dumps(missing_data_reports))
                            missing_data_reports = {}
                            report_i += 1
                            

                    except Exception as e:
                        missing_data_reports[curr_path] = {"proccesing_error": str(e), "error": True}
                        print(e)
                        import traceback
                        print(traceback.format_exc())
                    
                    curr_path = sampler.add_sample()
                    if not curr_path:
                        missing_data_reports[curr_path] = {"proccesing_error": "not enough valid samples", "error": True}
                        print("not enough valid samples")
                        break_out = True
                        break

        # write ranges to disk
        save_range_tracker(ranges, "./normalize_ranges.json")
        print("sample complete")

        # write missing data to disk
        with open("./missing_data_found_normalize.json", 'w') as f:
            f.write(json.dumps(missing_data_reports))
    
    print("completed")

