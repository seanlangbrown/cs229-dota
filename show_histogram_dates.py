import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import numpy as np

def plot_timestamp_histogram(timestamp_counts):
    """
    Create a histogram of timestamp counts, excluding 'unknown' key
    
    Args:
        timestamp_counts (dict): Dictionary with timestamp strings as keys and counts as values
    
    Returns:
        tuple: (total_count, unknown_count)
    """
    # Convert to pandas Series for easier manipulation
    series = pd.Series(timestamp_counts)
    
    # Separate known and unknown counts
    known_counts = series.drop('unknown', errors='ignore')
    unknown_count = timestamp_counts.get('unknown', 0)
    
    # Convert index to datetime for proper sorting
    known_counts.index = pd.to_datetime(known_counts.index)
    
    # Sort by date
    known_counts = known_counts.sort_index()
    
    # Calculate total count of known timestamps
    total_count = known_counts.sum()
    
    # Create plot
    plt.figure(figsize=(15, 6))
    known_counts.plot(kind='bar', edgecolor='black')
    plt.title('Timestamp Distribution')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Print totals
    print(f"Total count (excluding unknown): {total_count}")
    print(f"Unknown count: {unknown_count}")
    
    plt.show()
    
    return total_count, unknown_count

def plot_stacked_timestamp_histogram(no_tower_kills, with_tower_kills):
    """
    Create a stacked histogram of timestamp counts for two datasets
    
    Args:
        no_tower_kills (dict): Dictionary with no tower kills counts
        with_tower_kills (dict): Dictionary with tower kills counts
    
    Returns:
        tuple: (total_no_tower_kills, total_with_tower_kills, unknown_counts)
    """
    # Convert to pandas Series and remove 'unknown'
    no_tower_series = pd.Series(no_tower_kills).drop('unknown', errors='ignore')
    with_tower_series = pd.Series(with_tower_kills).drop('unknown', errors='ignore')
    
    # Get unknown counts
    no_tower_unknown = no_tower_kills.get('unknown', 0)
    with_tower_unknown = with_tower_kills.get('unknown', 0)
    
    # Convert index to datetime and sort
    no_tower_series.index = pd.to_datetime(no_tower_series.index)
    with_tower_series.index = pd.to_datetime(with_tower_series.index)
    
    # Convert to Unix timestamps (milliseconds / 1000)
    def to_unixtime(dt):
        return int(dt.timestamp() * 1000) // 1000
    
    # Create new series with Unix timestamp keys
    no_tower_unix = pd.Series(
        no_tower_series.values, 
        index=[to_unixtime(idx) for idx in no_tower_series.index]
    )
    with_tower_unix = pd.Series(
        with_tower_series.values, 
        index=[to_unixtime(idx) for idx in with_tower_series.index]
    )
    
    # Align series on common index
    combined_unix_index = sorted(set(no_tower_unix.index) | set(with_tower_unix.index))
    no_tower_aligned = no_tower_unix.reindex(combined_unix_index, fill_value=0)
    with_tower_aligned = with_tower_unix.reindex(combined_unix_index, fill_value=0)
    
    # Calculate total counts
    total_no_tower_kills = no_tower_aligned.sum()
    total_with_tower_kills = with_tower_aligned.sum()
    
    # Create stacked bar plot
    plt.figure(figsize=(20, 8))
    
    plt.bar(
        range(len(combined_unix_index)), 
        no_tower_aligned, 
        label=f'No Tower Kills (Total: {total_no_tower_kills:,})', 
        color='red', 
        edgecolor='red'
    )
    plt.bar(
        range(len(combined_unix_index)), 
        with_tower_aligned, 
        bottom=no_tower_aligned, 
        label=f'Yes Tower Kills (Total: {total_with_tower_kills:,})', 
        color='blue', 
        edgecolor='blue'
    )
    
    # Customize plot
    plt.title('Timestamp Distribution: Tower Kills Comparison', fontsize=16)
    plt.xlabel('Match Start Time', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Reduce number of x-axis labels
    num_labels = min(6, len(combined_unix_index))
    label_indices = [i * (len(combined_unix_index) // num_labels) for i in range(num_labels)]
    label_indices[-1] = len(combined_unix_index) - 1  # Ensure last label is included
    
    plt.xticks(
        label_indices, 
        [combined_unix_index[i] for i in label_indices], 
        rotation=45, 
        ha='right'
    )
    
    plt.legend(loc='upper left')
    plt.tight_layout()

    max_no_tower_kills = np.max(list(set(no_tower_unix.index))) # => 1607731200
    max_tower_kills = np.max(list(set(with_tower_unix.index)))
    
    # Print detailed totals
    print(f"Total No Tower Kills (excluding unknown): {total_no_tower_kills:,}")
    print(f"Unknown No Tower Kills: {no_tower_unknown:,}")
    print(f"Total Yes Tower Kills (excluding unknown): {total_with_tower_kills:,}")
    print(f"Unknown Yes Tower Kills: {with_tower_unknown:,}")
    print(f"Grand Total: {total_no_tower_kills + total_with_tower_kills:,}")
    print(f"Maximum time with no tower kills: {max_no_tower_kills}")
    print(f"Maximum time with tower kills: {max_tower_kills}")

    # Save the plot
    plt.savefig('tower_kills_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return (total_no_tower_kills, total_with_tower_kills, 
            no_tower_unknown, with_tower_unknown)

def find_significant_dropoff(timestamp_counts):
    # Convert to pandas Series and remove 'unknown'
    series = pd.Series(timestamp_counts).drop('unknown', errors='ignore')
    
    # Convert index to datetime and sort
    series.index = pd.to_datetime(series.index)
    
    # Convert to Unix timestamps (milliseconds / 1000)
    def to_unixtime(dt):
        return int(dt.timestamp() * 1000) // 1000
    
    # Create new series with Unix timestamp keys
    series_unix = pd.Series(
        series.values, 
        index=[to_unixtime(idx) for idx in series.index]
    )
    
    # Ensure sorted by timestamp
    series_unix = series_unix.sort_index()
    
       # Ensure sorted by timestamp
    series_unix = series_unix.sort_index()
    
    # Calculate absolute change between consecutive timestamps
    absolute_change = series_unix.diff().abs()

    area_of_interest = absolute_change.iloc[2100:]
    
    # Find the index of the largest absolute change
    max_change_index = area_of_interest.argmax() + 2100

    print(max_change_index)
    
    # # Get details of the change
    max_change_timestamp = absolute_change.index[int(max_change_index)]
    # max_change_value = absolute_change[max_change_index]
    
    # Visualize to confirm
    plt.figure(figsize=(15,6))
    absolute_change.plot(kind='line')
    plt.title('No Tower Kills Histograme with Cutoff Highlighted')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.axvline(x=max_change_timestamp, color='r', linestyle='--') # type: ignore
    plt.show()
    
    print(f"Largest absolute change at: {max_change_timestamp}") # 1265932800
    # print(f"Absolute change value: {max_change_value}")
    
    # return max_change_timestamp, max_change_value

def main():
    # Read JSON files
    try:
        filename = './statistics/count_matches_by_day_0.json'
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Extract the two datasets
        with_tower_kills = data.get('yes_tower_kills', {})

        filename = './statistics/count_matches_by_day_1.json'
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Extract the two datasets
        no_tower_kills = data.get('no_tower_kills', {})
    except FileNotFoundError:
        print(f"File not found: {filename}") # type: ignore
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {filename}") # type: ignore
        sys.exit(1)
    
    # Call the function to plot and print counts
    plot_stacked_timestamp_histogram(no_tower_kills, with_tower_kills)

    #Find the dropoff in no_tower_kills
    find_significant_dropoff(no_tower_kills)

if __name__ == '__main__':
    main()

# def main():
#     # Check if filename is provided as command-line argument
#     if len(sys.argv) < 2:
#         print("Please provide the JSON file path as a command-line argument")
#         sys.exit(1)
    
#     # Get filename from command-line argument
#     filename = sys.argv[1]
    
#     # Read JSON file
#     try:
#         with open(filename, 'r') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"File not found: {filename}")
#         sys.exit(1)
#     except json.JSONDecodeError:
#         print(f"Invalid JSON in file: {filename}")
#         sys.exit(1)

#     ytk = data["yes_tower_kills"]
#     ntk = data["no_tower_kills"]
    
#     # Call the function to plot and print counts
#     plot_timestamp_histogram(ytk)

# if __name__ == '__main__':
#     main()