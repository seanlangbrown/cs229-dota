#!/usr/bin/env python3

from sample_statistics import get_normalization_parameters, count_specific_s3_paths, list_all_match_files
from normalize import load_range_tracker, summarize_range_tracker
import sys

if __name__ == "__main__":
    try:
        mode = sys.argv[1]

        if mode == "list_matches":
            list_all_match_files()

        elif mode == "summary":
            print('summarizing range tracker')
            filename = sys.argv[2]

            if filename is None:
                filename = './normalize_ranges.json'
            rt = load_range_tracker(filename)
            summarize_range_tracker(rt)

        elif mode == "count":
            i = sys.argv[2]
            _, _2 = count_specific_s3_paths(i=i)
            print("completed")

        else:
            n = int(sys.argv[2])
            get_normalization_parameters(n)
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        print(stack_trace)
        exit(1)
