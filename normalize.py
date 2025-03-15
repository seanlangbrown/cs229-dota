import numpy as np
import pandas as pd
import json
import math
from cs229_utils import print_full_df

def toType(s):
    try:
        if '.' in s:
            return float(s)
        else:
            return int(s)
    except Exception:
        if s == 'True' or s == 'False':
            return bool(s)
        else:
            return s

class RangeTracker:
    def __init__(self, initial_df=None):
        self.extremes = {
            'max': {},
            'min': {},
            'std': {},
            'count': {},
            'sum': {},
            'sum_squared': {}
        }
        self.count = 0
        self.merged_columns = {}
        if initial_df:
            self._update_extremes(initial_df)
            self.count = 1

    def _update_extremes(self, df):
        for col in df.columns:
            # Filter out NaN values before processing
            valid_values = df[col].dropna()
            
            if len(valid_values) == 0:
                continue
            
            if pd.api.types.is_numeric_dtype(valid_values):
                if col not in self.extremes['max']:
                    self.extremes['max'][col] = valid_values.max()
                    self.extremes['min'][col] = valid_values.min()
                    self.extremes['count'][col] = len(valid_values)
                    self.extremes['sum'][col] = valid_values.sum()
                    self.extremes['sum_squared'][col] = np.sum(valid_values**2)
                else:
                    current_count = self.extremes['count'][col]
                    new_count = current_count + len(valid_values)
                    
                    # Update max and min
                    self.extremes['max'][col] = max(self.extremes['max'][col], valid_values.max())
                    self.extremes['min'][col] = min(self.extremes['min'][col], valid_values.min())
                    
                    # Update sum and sum of squares
                    self.extremes['sum'][col] += valid_values.sum()
                    self.extremes['sum_squared'][col] += np.sum(valid_values**2)
                    
                    # Update count
                    self.extremes['count'][col] = new_count
            
            elif pd.api.types.is_bool_dtype(valid_values):
                if col not in self.extremes['max']:
                    self.extremes['max'][col] = True if valid_values.any() else False
                    self.extremes['min'][col] = False if not valid_values.all() else True
            
            elif pd.api.types.is_string_dtype(valid_values):
                if col not in self.extremes['max']:
                    self.extremes['max'][col] = valid_values.max()
                    self.extremes['min'][col] = valid_values.min()
                else:
                    self.extremes['max'][col] = max(self.extremes['max'][col], valid_values.max())
                    self.extremes['min'][col] = min(self.extremes['min'][col], valid_values.min())

    def update(self, new_df):
        self._update_extremes(new_df)
        self.count+=1

    def get_count(self, column):
        return self.extremes['count'][column]
        
    def get_total_count(self):
        if len(self.list_columns()) == 0:
            return 0
        return np.max([self.get_count(c) for c in self.list_columns()])
    
    def get_min_count(self):
        return np.min([self.get_count(c) for c in self.list_columns()])
    
    def get_column_count(self):
        return len(self.extremes['count'])
    
    def list_columns(self):
        return self.extremes['max'].keys()

    def get_max(self, column):
        return self.extremes['max'].get(column)

    def get_min(self, column):
        return self.extremes['min'].get(column)

    def get_std(self, column):
        if column not in self.extremes['count'] or self.extremes['count'][column] <= 1:
            return None
        
        count = self.extremes['count'][column]
        total_sum = self.extremes['sum'][column]
        sum_squared = self.extremes['sum_squared'][column]
        
        # Calculate variance using the computational formula
        variance = (sum_squared / count) - (total_sum / count)**2
        return math.sqrt(max(0, variance))
    
    def get_mean(self, column):
        if column not in self.extremes['count'] or self.extremes['count'][column] <= 1:
            return None
        
        count = self.extremes['count'][column]
        total_sum = self.extremes['sum'][column]

        return total_sum / count
    
    def get_column_table(self, column):
        df = pd.DataFrame(
            data={
                'column': column,
                'min': self.get_min(column), 
                'max': self.get_max(column), 
                'mean': self.get_mean(column),
                'std': self.get_std(column),
                'count': self.get_count(column)
            },
            index=[column]
        )
        return df
    
    def get_table(self):
        dfs = [self.get_column_table(c) for c in self.list_columns()]
        result = pd.concat(dfs)
        return result
    
    def merge_columns(self, columns_to_merge, new_column_name, remove_old_columns=True):
        if not all(col in self.list_columns() for col in columns_to_merge):
            raise ValueError("Not all specified columns exist")
        
        # Merge numeric statistics
        for stat in ['max', 'min', 'count', 'sum', 'sum_squared']:
            self.extremes[stat][new_column_name] = sum(
                self.extremes[stat].get(col, 0) for col in columns_to_merge
            )
        
        # Store mapping of merged columns (reversed)
        for col in columns_to_merge:
            self.merged_columns[col] = new_column_name
        
        # Remove old columns if specified
        if remove_old_columns:
            for col in columns_to_merge:
                for stat in ['max', 'min', 'count', 'sum', 'sum_squared']:
                    del self.extremes[stat][col]
        
        return self

    def z_score_normalize(self, df, columns=None, use_merged=True):
        """
        Applies z-score normalization to every numeric column in the dataframe, or all columns specified
        """
        if columns is None:
            columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        normalized_df = df.copy()
        for col in columns:
            # Check for merged columns if use_merged is True
            if use_merged and col in self.merged_columns:
                merged_col = self.merged_columns[col]
            else:
                merged_col = col
            
            mean = self.get_mean(merged_col)
            std = self.get_std(merged_col)
            
            if mean is not None and std is not None and std > 0:
                normalized_df[col] = (df[col] - mean) / std
        
        return normalized_df

    def min_max_normalize(self, df, columns=None, use_merged=True):
        """
        Applies min-max normalization to every numeric column in the dataframe, or all columns specified
        """
        if columns is None:
            columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        normalized_df = df.copy()
        for col in columns:
            # Check for merged columns if use_merged is True
            if use_merged and col in self.merged_columns:
                merged_col = self.merged_columns[col]
            else:
                merged_col = col
            
            min_val = self.get_min(merged_col)
            max_val = self.get_max(merged_col)
            
            if min_val is not None and max_val is not None and min_val != max_val:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return normalized_df

    def serialize(self):
        return json.dumps({
            'max': {k: str(v) for k, v in self.extremes['max'].items()},
            'min': {k: str(v) for k, v in self.extremes['min'].items()},
            'count': {k: int(v) for k, v in self.extremes['count'].items()},
            'sum': {k: str(v) for k, v in self.extremes['sum'].items()},
            'sum_squared': {k: str(v) for k, v in self.extremes['sum_squared'].items()}
        }, allow_nan=False)

    @classmethod
    def deserialize(cls, serialized_data):
        data = json.loads(serialized_data)
        tracker = cls.__new__(cls)
        tracker.extremes = {
            'max': {k: (toType(v)) for k, v in data['max'].items()},
            'min': {k: (toType(v)) for k, v in data['min'].items()},
            'count': data['count'],
            'sum': {k: float(v) for k, v in data['sum'].items()},
            'sum_squared': {k: float(v) for k, v in data['sum_squared'].items()},
            'std': {}
        }
        tracker.merged_columns = {}
        tracker.count = tracker.get_total_count()
        return tracker
    
   
def load_range_tracker(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return RangeTracker.deserialize(data)

def save_range_tracker(range_tracker, file_path):
    with open(file_path, 'w') as f:
        f.write(range_tracker.serialize())


def summarize_range_tracker(range_tracker):
    print(f"df count: {range_tracker.get_total_count()}")
    print(f"column count: {range_tracker.get_column_count()}")
    print(f"smallest count: {range_tracker.get_min_count()}")
    print_full_df(range_tracker.get_table())


   
