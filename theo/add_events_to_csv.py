"""
Script to add events from JSONL file to nose_cap labeled CSV without duplicates
Uses the same preprocessing pipeline as the notebook
"""
import json
import pandas as pd
import numpy as np
import pytz
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import glob
import os

class JSONL_DataProcessor:
    """
    reads and processes JSONL data from pulse device into usable dataframes for analysis
    """

    def __init__(self, file_path):
        """
        Initializes the DataProcessor with a file path to the JSONL data.

        Args:
            file_path : path to JSONL data file from Pulse device
        """
        
        # file path to the JSON data file
        self.file_path = file_path

        # read, unpack, clean, and classify the data
        print(f"Reading data from {self.file_path}...")
        self.data = self._read_data() # read the data from the file path
        self.all_packets_df = self._unpack_data(self.data) # unpack the data into a dataframe
        self.all_packets_df = self._clean_data(self.all_packets_df) # clean the data 
        self.all_packets_df = self._classify_data_with_date_range_labels(self.all_packets_df) # add date range labels to the data
        self.all_packets_df = self._classify_data_with_operational_phase(self.all_packets_df) # add start up or production labels to the data, interested in analysis on production data only 
        
        # done processing data, return some basic information 
        print("\nData processing complete.")
        print(f"Total number of cycles: {self.all_packets_df.shape[0]}")
        print(f"DataFrame Columns: {self.all_packets_df.columns.tolist()}")

    def _read_data(self):
        """
        Reads the JSONL data from the specified file path.

        Returns:
            data : dataframe with the read in file from the Pulse device
        """
        print("\nrunning _read_data()...")

        data_list = []
        with open(self.file_path, 'r') as file:
            for line in file:
                # parse each line as a separate JSON object
                data_list.append(json.loads(line))

        # normalize the list of JSON objects
        data = pd.json_normalize(data_list)

        return data

    def _unpack_data(self, data):
        """
        Unpacks the JSONL data into a DataFrame with specific columns

        Args:
            data (DataFrame): The DataFrame containing the JSONL data from Pulse device

        Returns:
            unpacked_data (DataFrame): DataFrame with unpacked data containing specific columns
        """
        print("\nrunning _unpack_data()...")

        unpacked_data_list = []

        for packets in data['data.fftPackets']:
            # extract open and close events
            open_event = packets[0]
            close_event = packets[1]

            # create a dictionary to store the combined data
            combined_data = {
                'open_end_time': open_event['end_time'],
                'open_start_time': open_event['start_time'],
                'open_fft_accum': open_event['fft_accum'],
                'open_fft_accum_count': open_event['fft_accum_count'],
                'close_end_time': close_event['end_time'],
                'close_start_time': close_event['start_time'],
                'close_fft_accum': close_event['fft_accum'],
                'close_fft_accum_count': close_event['fft_accum_count']
            }
            
            # append combined data to the list
            unpacked_data_list.append(combined_data)

        # convert unpacked_data_list into all_packets_df DataFrame
        all_packets_df = pd.DataFrame(unpacked_data_list)

        # flip the order of the rows and reset index, so time and cycle count are in order
        all_packets_df = all_packets_df[::-1].reset_index(drop=True)

        # add cycle count column
        all_packets_df['cycle_count'] = all_packets_df.index

        # convert time columns to datetime - explicitly specify UTC first, then convert to my local timezone (Chicago)
        all_packets_df['open_end_time'] = pd.to_datetime(all_packets_df['open_end_time'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        all_packets_df['open_start_time'] = pd.to_datetime(all_packets_df['open_start_time'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        all_packets_df['close_end_time'] = pd.to_datetime(all_packets_df['close_end_time'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        all_packets_df['close_start_time'] = pd.to_datetime(all_packets_df['close_start_time'], unit='s', utc=True).dt.tz_convert('America/Chicago')

        # add cycle time column
        all_packets_df['cycle_time'] = (all_packets_df['close_end_time'] - all_packets_df['open_start_time']).dt.total_seconds()

        return all_packets_df
    
    def _clean_data(self, all_packets_df):
        """
        Cleans the DataFrame by removing:
        1 - rows with buggy timestamps (abnormally large cycle times)
        2 - rows with 0 values for accum count
        3 - rows with unusual magnitudes for FFT accum (using IQR method)

        Args:
            all_packets_df (DataFrame): The DataFrame containing the unpacked data

        Returns:
            all_packets_df (DataFrame): DataFrame with cleaned data and cycle count
        """
        print("\nrunning _clean_data()...")

        print(f"dataset size before cleaning: {all_packets_df.shape}")

        ## remove rows with buggy timestamps (abnormally large cycle times)
        print("\n1 - Removing rows with abnormal start/end timestamps...")

        # calculate the mean, median, and std of cycle time
        mean_cycle_time = all_packets_df['cycle_time'].mean()
        median_cycle_time = all_packets_df['cycle_time'].median()
        std_cycle_time = all_packets_df['cycle_time'].std()
        lower_bound = median_cycle_time / 2
        upper_bound = median_cycle_time * 2

        # print the statistics
        print(f"\nCycle Time Statistics:")
        print(f"mean cycle time: {mean_cycle_time}")
        print(f"median cycle time: {median_cycle_time}")
        print(f"std cycle time: {std_cycle_time}")
        print(f"Cycle Time Threshold: {lower_bound} - {upper_bound}")

        # identify rows with cycle time outside the threshold and print them
        outliers = all_packets_df[(all_packets_df['cycle_time'] < lower_bound) | (all_packets_df['cycle_time'] > upper_bound)]
        print(f"\nOutliers: {len(outliers)}")

        # remove cycle time outliers from all_packets_df
        all_packets_df = all_packets_df[(all_packets_df['cycle_time'] >= lower_bound) & (all_packets_df['cycle_time'] <= upper_bound)].reset_index(drop=True)


        ## remove rows with 0 values for accum count
        print("\n2 - Removing rows with 0 values for accum count...")
        all_packets_df = all_packets_df[(all_packets_df['open_fft_accum_count'] != 0) & (all_packets_df['close_fft_accum_count'] != 0)].reset_index(drop=True) 


        ## remove rows with unusual magnitudes for fft accum
        print("\n3 - Removing rows with unusual magnitudes for FFT accum...")

        # Calculate summary statistics for each FFT list
        all_packets_df['open_fft_max'] = all_packets_df['open_fft_accum'].apply(lambda x: max(x) if len(x) > 0 else 0)
        all_packets_df['close_fft_max'] = all_packets_df['close_fft_accum'].apply(lambda x: max(x) if len(x) > 0 else 0)

        # Calculate IQR bounds for max values
        Q1_open = all_packets_df['open_fft_max'].quantile(0.25)
        Q3_open = all_packets_df['open_fft_max'].quantile(0.75)
        IQR_open = Q3_open - Q1_open
        lower_bound_open = Q1_open - 3 * IQR_open
        upper_bound_open = Q3_open + 3 * IQR_open

        Q1_close = all_packets_df['close_fft_max'].quantile(0.25)
        Q3_close = all_packets_df['close_fft_max'].quantile(0.75)
        IQR_close = Q3_close - Q1_close
        lower_bound_close = Q1_close - 3 * IQR_close
        upper_bound_close = Q3_close + 3 * IQR_close

        print(f"\nFFT Max Thresholds:")
        print(f"Open FFT Max: {lower_bound_open:.2f} - {upper_bound_open:.2f}")
        print(f"Close FFT Max: {lower_bound_close:.2f} - {upper_bound_close:.2f}")

        # Identify FFT outliers
        fft_outliers = all_packets_df[
            (all_packets_df['open_fft_max'] < lower_bound_open) | 
            (all_packets_df['open_fft_max'] > upper_bound_open) |
            (all_packets_df['close_fft_max'] < lower_bound_close) | 
            (all_packets_df['close_fft_max'] > upper_bound_close)
        ]

        print(f"\nFFT Outliers found: {len(fft_outliers)}")

        # Remove FFT outliers
        all_packets_df = all_packets_df[
            (all_packets_df['open_fft_max'] >= lower_bound_open) & 
            (all_packets_df['open_fft_max'] <= upper_bound_open) &
            (all_packets_df['close_fft_max'] >= lower_bound_close) & 
            (all_packets_df['close_fft_max'] <= upper_bound_close)
        ].reset_index(drop=True)

        # Drop the temporary columns if you don't need them
        all_packets_df = all_packets_df.drop(['open_fft_max', 'close_fft_max'], axis=1)


        # df length after cleaning
        print(f"dataset size after cleaning: {all_packets_df.shape}")

        return all_packets_df

    def _classify_data_with_date_range_labels(self, all_packets_df):
        """
        adds data range labels for week and day to each datapoint
        allows us to filter datapoints by specific dates and times

        Args:
            all_packets_df (DataFrame): cleaned dataframe 

        Returns:
            all_packets_df (DataFrame): cleaned dataframe with date range labels
        """
        print("\nrunning _classify_data_with_date_range_labels()...")

        ## all_packets_df - add date range labels for open and close events (helps analyze time series data)

        # Define date ranges dynamically
        timezone = pytz.timezone('America/Chicago')

        # Get the date range from the data
        min_date = all_packets_df['close_end_time'].min()
        max_date = all_packets_df['close_end_time'].max()
        
        print(f"Data date range: {min_date} to {max_date}")

        # Generate day ranges dynamically based on data
        day_ranges = []
        current_date = min_date.date()
        end_date = max_date.date()
        
        while current_date <= end_date:
            start_ts = timezone.localize(pd.Timestamp(current_date))
            end_ts = timezone.localize(pd.Timestamp(current_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            day_ranges.append((start_ts, end_ts, str(current_date)))
            current_date = (pd.Timestamp(current_date) + pd.Timedelta(days=1)).date()

        # For week ranges, group by week
        week_ranges = []
        current_date = min_date.date()
        week_start = current_date
        week_num = 1
        
        while current_date <= end_date:
            # Check if we've completed a week (7 days) or reached the end
            days_since_week_start = (current_date - week_start).days
            if days_since_week_start >= 7 or current_date == end_date:
                start_ts = timezone.localize(pd.Timestamp(week_start))
                end_ts = timezone.localize(pd.Timestamp(current_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                week_ranges.append((start_ts, end_ts, f'week {week_num}'))
                if current_date < end_date:
                    week_start = (pd.Timestamp(current_date) + pd.Timedelta(days=1)).date()
                    week_num += 1
            current_date = (pd.Timestamp(current_date) + pd.Timedelta(days=1)).date()

        ## Generalized function to label date ranges
        def label_date_range(date, ranges):
            for start, end, label in ranges:
                if start <= date <= end:
                    return label
            return 'Other'  # Default label if no range matches

        ## Apply the function to create a new column with labels (change 'day_ranges' to 'week_ranges' for weekly labels)
        all_packets_df['date_range'] = all_packets_df['close_end_time'].apply(lambda x: label_date_range(x, day_ranges))
        all_packets_df['week_range'] = all_packets_df['close_end_time'].apply(lambda x: label_date_range(x, week_ranges))

        ## Generate a new column normalized time from start to end of dataset -> allows us to generate a heatmap of sequence of events in pca
        # this is a more generalized version of 'date_range'
        # normalizes the time from 0 to 1 where 0 is the start of the dataset and 1 is the end of the dataset
        time_scaler = MinMaxScaler()    
        all_packets_df['normalized_open_time'] = time_scaler.fit_transform((all_packets_df['open_end_time'] - all_packets_df['open_end_time'].min()).dt.total_seconds().values.reshape(-1, 1))
        all_packets_df['normalized_close_time'] = time_scaler.fit_transform((all_packets_df['close_end_time'] - all_packets_df['close_end_time'].min()).dt.total_seconds().values.reshape(-1, 1))

        return all_packets_df

    def _classify_data_with_operational_phase(self, all_packets_df):
        """
        adds an additional column to the DataFrame to classify each cycle as 'startup' or 'production'.
        start up datapoints are identified by the duration of time since the last datapoint
        
        this is important because for analysis we are only interested in production data, not start up data
        start up data is frequently noisy 
        this also helps filter out outliers that are not part of the production cycle (i.e 1-2 random datapoints after a production run that are clearly not part of anything significant)

        Args:
            all_packets_df (DataFrame): cleaned dataframe

        Returns:
            all_packets_df (DataFrame): cleaned dataframe with operational phase classification

        """
        print("\nrunning _classify_data_with_operational_phase...")

        # min gap between datapoint n and n-1 to consider as a run start (minutes)
        gap_threshold_minutes = 30

        # duration to consider for startup period (minutes)
        startup_duration_minutes = 45

        # calculate the gap between consecutive cycles
        all_packets_df['gap_between_cycles_minutes'] = all_packets_df['open_start_time'].diff().dt.total_seconds() / 60  # calculate the gap in minutes

        # identify the start of runs based on the gap threshold
        all_packets_df['is_run_start'] = all_packets_df['gap_between_cycles_minutes'] > gap_threshold_minutes
        all_packets_df.loc[all_packets_df.index[0], 'is_run_start'] = False  # First cycle is not a run start

        # assign run numbers 
        all_packets_df['run_number'] = all_packets_df['is_run_start'].cumsum()

        # calculate time since run start for each datapoint using run number, allows us to label cycles as startup or production
        run_start_times = all_packets_df[all_packets_df['is_run_start']].set_index('run_number')['open_start_time']
        all_packets_df['run_start_time'] = all_packets_df['run_number'].map(run_start_times)
        all_packets_df['minutes_since_run_start'] = (all_packets_df['open_start_time'] - all_packets_df['run_start_time']).dt.total_seconds() / 60

        # classify operational phases based on minutes since start
        def classify_operational_phase(minutes_since_start):
            if pd.isna(minutes_since_start):
                return 'start up'
            if minutes_since_start < startup_duration_minutes:
                return 'start up'
            else:
                return 'production'
            
        # run function to classify
        all_packets_df['operational_phase'] = all_packets_df['minutes_since_run_start'].apply(classify_operational_phase)

        return all_packets_df
    
class FeatureEngineering:
    """
    Extract features from all_packets_df for machine learning models
    """

    def __init__(self, all_packets_df):
        self.all_packets_df = all_packets_df
        
        # Initialize feature DataFrames as instance variables
        self.all_features = None
        self.open_features = None
        self.close_features = None
        
        # Initialize normalized feature DataFrames
        self.all_features_normalized = None
        self.open_features_normalized = None
        self.close_features_normalized = None
        
        # Store scalers for potential inverse transformation
        self.all_features_scaler = None
        self.open_features_scaler = None
        self.close_features_scaler = None
        
        # Track processing state
        self.features_extracted = False
        self.features_normalized = False

    def extract_features(self):
        """
        Extract features from all_packets_df for machine learning models.

        1 - computes features and saves them to open/close/combined feature dataframes
        2 - adds back timestamps and time labels for sequential analysis to the three feature dataframes 
        3 - stores results in instance variables and returns them

        Returns:
            tuple: (all_features, open_features, close_features) DataFrames
        """

        print("Extracting features from all_packets_df...")

        # 1 - compute and save features 
        feature_list = []
        open_feature_list = []
        close_feature_list = []
        
        # Iterate through each row in the DataFrame to compute features_df
        print("Computing features for each packet...")
        for _, row in self.all_packets_df.iterrows():
            open_features_dict = self.compute_features(row['open_fft_accum'], row['open_fft_accum_count'], 'open')
            close_features_dict = self.compute_features(row['close_fft_accum'], row['close_fft_accum_count'], 'close')

            # Store individual feature dictionaries
            open_feature_list.append(open_features_dict)
            close_feature_list.append(close_features_dict)
            
            # Combine for all_features
            combined_features = {**open_features_dict, **close_features_dict}
            feature_list.append(combined_features)

        # Create DataFrames from lists of dictionaries
        self.open_features = pd.DataFrame(open_feature_list)
        self.close_features = pd.DataFrame(close_feature_list)
        self.all_features = pd.DataFrame(feature_list)

        print("...Features computed successfully.")

        print("Adding timestamps and time labels for sequential analysis...")

        # 2 - add back timestamps and time labels for sequential analysis 
        timestamp_columns = ['open_start_time', 'open_end_time', 'operational_phase']
        
        # Check which columns actually exist in the source DataFrame
        available_columns = [col for col in timestamp_columns if col in self.all_packets_df.columns]
        
        if available_columns:
            # Add to open_features
            for col in available_columns:
                self.open_features[col] = self.all_packets_df[col].values
            
            # Add to close_features (include close times if available)
            close_timestamp_columns = available_columns.copy()
            if 'close_start_time' in self.all_packets_df.columns:
                close_timestamp_columns.append('close_start_time')
            if 'close_end_time' in self.all_packets_df.columns:
                close_timestamp_columns.append('close_end_time')
            
            available_close_columns = [col for col in close_timestamp_columns if col in self.all_packets_df.columns]
            for col in available_close_columns:
                self.close_features[col] = self.all_packets_df[col].values
            
            # Add all available timestamp columns to all_features
            all_timestamp_columns = ['open_start_time', 'open_end_time', 
                                   'close_start_time', 'close_end_time',
                                   'cycle_time', 'operational_phase']
            available_all_columns = [col for col in all_timestamp_columns if col in self.all_packets_df.columns]
            for col in available_all_columns:
                self.all_features[col] = self.all_packets_df[col].values

        print("...Timestamps and time labels added successfully.")

        # Mark features as extracted
        self.features_extracted = True
        
        print("Features extracted successfully.")
        # 3 - return the three DataFrames
        return self.all_features, self.open_features, self.close_features

    def compute_features(self, fft_data, fft_accum_count, prefix):
        """
        Helper function to compute features, called from extract_features method.

        Args:
            fft_data: subset of the all_packets_df containing fft data for either open or close events
            fft_accum_count: subset of the all_packets_df containing fft accum count for either open or close events
            prefix: prefix to use for feature names, either 'open' or 'close'
            
        Returns:
            dict: Dictionary containing computed features with appropriate column names.
        """
        # Normalize the FFT data with the accumulation count
        fft_normalized = [value / fft_accum_count for value in fft_data]
        
        # define sampling frequency for PSD 
        fs = 100  # hz 

        # Compute Power Spectral Density (PSD)
        psd = np.square(fft_normalized) / (len(fft_normalized) * fs)

        # Frequency indices (assuming uniform sampling)
        freqs = np.arange(len(fft_normalized))

        # Compute additional spectral features
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
        spectral_flatness = np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd) if np.mean(psd) > 0 else 0  # Adding epsilon to avoid log(0)

        # Extract features 
        features = {
            f'{prefix}_raw_fft_normalized': fft_normalized,  # raw fft data normalized against fft accum count
            f'{prefix}_fft_std': np.std(fft_normalized),
            f'{prefix}_psd_mean': np.mean(psd),
            f'{prefix}_spectral_entropy': entropy(psd),  # (lower values -> ordered, power is concentrated in a few bins)
            f'{prefix}_spectral_bandwidth': spectral_bandwidth,  # (higher values -> more spread out)
            f'{prefix}_spectral_flatness': spectral_flatness,  # (higher values -> more noise, lower values -> more tonal)
        }
        
        return features

def identify_duplicates(new_df, existing_df, timestamp_col='open_start_time'):
    """Identify duplicate rows based on timestamp"""
    if timestamp_col not in new_df.columns or timestamp_col not in existing_df.columns:
        print(f"Warning: Timestamp column '{timestamp_col}' not found. Trying alternative columns...")
        # Try alternative timestamp columns
        alt_cols = ['open_start_time', 'close_end_time', 'open_end_time', 'close_start_time']
        for alt_col in alt_cols:
            if alt_col in new_df.columns and alt_col in existing_df.columns:
                timestamp_col = alt_col
                print(f"Using '{alt_col}' for duplicate detection")
                break
        else:
            print("No suitable timestamp column found. Skipping duplicate detection.")
            return pd.DataFrame()
    
    # Convert timestamps to comparable format
    try:
        # Handle timezone-aware timestamps
        if hasattr(new_df[timestamp_col].dtype, 'tz') and new_df[timestamp_col].dtype.tz is not None:
            new_timestamps = pd.to_datetime(new_df[timestamp_col]).dt.tz_localize(None)
        else:
            new_timestamps = pd.to_datetime(new_df[timestamp_col])
            
        if hasattr(existing_df[timestamp_col].dtype, 'tz') and existing_df[timestamp_col].dtype.tz is not None:
            existing_timestamps = pd.to_datetime(existing_df[timestamp_col]).dt.tz_localize(None)
        else:
            existing_timestamps = pd.to_datetime(existing_df[timestamp_col])
    except Exception as e:
        print(f"Error converting timestamps: {e}. Using string comparison.")
        new_timestamps = new_df[timestamp_col].astype(str)
        existing_timestamps = existing_df[timestamp_col].astype(str)
    
    # Find duplicates
    duplicates = new_df[new_timestamps.isin(existing_timestamps)]
    
    return duplicates

def normalize_series_to_naive(series):
    """Force a series to be timezone-naive datetime"""
    # First, ensure it's datetime
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            series = pd.to_datetime(series)
        except Exception:
            # If bulk conversion fails, might be due to mixed types
            return series

    # If it has timezone info, strip it
    if hasattr(series, 'dt') and series.dt.tz is not None:
        return series.dt.tz_localize(None)
    
    # If it's still object type (mixed naive/aware timestamps), handle row-by-row
    if series.dtype == 'object':
        # Try to identify if any elements are tz-aware
        def make_naive_scalar(x):
            if pd.isna(x): return x
            if hasattr(x, 'tzinfo') and x.tzinfo is not None:
                return x.replace(tzinfo=None)
            return x
        
        # It's faster to try converting to datetime with utc=True then localize None if possible,
        # but if we want to preserve local time without shifting, we need to be careful.
        # Simplest robust way for mixed bag:
        try:
            return series.apply(make_naive_scalar)
        except:
            pass

    return series

def merge_dataframes(new_df, existing_df, timestamp_col='open_start_time'):
    """Merge new data with existing data, removing duplicates"""
    print(f"\nMerging dataframes...")
    print(f"  Existing CSV rows: {len(existing_df)}")
    print(f"  New JSONL rows: {len(new_df)}")
    
    # Normalize timestamps in both dataframes BEFORE anything else
    if timestamp_col in new_df.columns:
        new_df[timestamp_col] = normalize_series_to_naive(new_df[timestamp_col])
    if timestamp_col in existing_df.columns:
        existing_df[timestamp_col] = normalize_series_to_naive(existing_df[timestamp_col])
    
    # Also normalize other timestamp columns to avoid future issues
    ts_cols = ['open_start_time', 'open_end_time', 'close_start_time', 'close_end_time']
    for col in ts_cols:
        if col in new_df.columns:
            new_df[col] = normalize_series_to_naive(new_df[col])
        if col in existing_df.columns:
            existing_df[col] = normalize_series_to_naive(existing_df[col])

    # Identify duplicates
    duplicates = identify_duplicates(new_df, existing_df, timestamp_col)
    print(f"  Duplicate rows found: {len(duplicates)}")
    
    # Remove duplicates from new data
    if len(duplicates) > 0 and timestamp_col in new_df.columns and timestamp_col in existing_df.columns:
        try:
            # With normalized timestamps, simple boolean indexing should work
            # But let's be extra safe with isin
            existing_timestamps = existing_df[timestamp_col].unique()
            new_df_filtered = new_df[~new_df[timestamp_col].isin(existing_timestamps)].copy()
        except Exception as e:
            print(f"Error filtering duplicates: {e}. Using string comparison.")
            new_timestamps = new_df[timestamp_col].astype(str)
            existing_timestamps = existing_df[timestamp_col].astype(str)
            new_df_filtered = new_df[~new_timestamps.isin(existing_timestamps)].copy()
    else:
        new_df_filtered = new_df.copy()
    
    print(f"  New unique rows to add: {len(new_df_filtered)}")
    
    # Ensure both dataframes have the same columns
    # Add missing columns to new_df_filtered with NaN values
    missing_cols = set(existing_df.columns) - set(new_df_filtered.columns)
    for col in missing_cols:
        new_df_filtered.loc[:, col] = np.nan
    
    # Reorder columns to match existing_df
    new_df_filtered = new_df_filtered[existing_df.columns]
    
    # Concatenate dataframes
    merged_df = pd.concat([existing_df, new_df_filtered], ignore_index=True)
    
    # Sort by timestamp if available
    if timestamp_col in merged_df.columns:
        # Final check to ensure the merged column is normalized
        merged_df[timestamp_col] = normalize_series_to_naive(merged_df[timestamp_col])
        merged_df = merged_df.sort_values(by=timestamp_col).reset_index(drop=True)
    
    print(f"  Final merged rows: {len(merged_df)}")
    
    return merged_df

def main():
    # File paths
    csv_file = "24-121_all_features_labels.csv"
    
    print("="*70)
    print("ADDING EVENTS FROM JSONL TO CSV")
    print("="*70)
    
    # Step 1: Find all JSONL files in the current directory
    jsonl_files = glob.glob("*.jsonl")
    
    if not jsonl_files:
        print("Error: No JSONL files found in the current directory")
        return
    
    jsonl_files.sort()  # Sort for consistent processing order
    print(f"\nFound {len(jsonl_files)} JSONL file(s):")
    for i, file in enumerate(jsonl_files, 1):
        print(f"  {i}. {file}")
    
    # Step 2: Process all JSONL files and combine their features
    all_combined_features = []
    
    for jsonl_file in jsonl_files:
        print("\n" + "="*70)
        print(f"PROCESSING: {jsonl_file}")
        print("="*70)
        
        try:
            # Process JSONL file using JSONL_DataProcessor
            print("\n" + "="*50)
            print("PROCESSING JSONL FILE")
            print("="*50)
            data_processor = JSONL_DataProcessor(jsonl_file)
            
            # Extract features using FeatureEngineering
            print("\n" + "="*50)
            print("FEATURE ENGINEERING")
            print("="*50)
            feature_engineer = FeatureEngineering(data_processor.all_packets_df)
            all_features, open_features, close_features = feature_engineer.extract_features()
            
            # Add useful data columns back in
            print("\nAdding missing date/time columns to feature DataFrames...")
            
            useful_columns = [
                'cycle_count', 'cycle_time', 'date_range', 'week_range',
                'normalized_open_time', 'normalized_close_time',
                'run_number', 'minutes_since_run_start', 'gap_between_cycles_minutes'
            ]
            
            available_columns = [col for col in useful_columns if col in data_processor.all_packets_df.columns]
            print(f"Available columns to add: {available_columns}")
            
            if available_columns:
                for col in available_columns:
                    all_features[col] = data_processor.all_packets_df[col].values
            
            # Normalize timezone-aware timestamps in all_features to timezone-naive for consistency
            timestamp_cols = ['open_start_time', 'open_end_time', 'close_start_time', 'close_end_time']
            for col in timestamp_cols:
                if col in all_features.columns:
                    if hasattr(all_features[col].dtype, 'tz') and all_features[col].dtype.tz is not None:
                        all_features.loc[:, col] = pd.to_datetime(all_features[col]).dt.tz_localize(None)
            
            print(f"\nall_features shape: {all_features.shape}")
            print(f"all_features columns: {list(all_features.columns)}")
            
            # Add to combined list
            all_combined_features.append(all_features)
            print(f"✅ Successfully processed {jsonl_file}")
            
        except Exception as e:
            print(f"❌ Error processing {jsonl_file}: {e}")
            print(f"Skipping this file and continuing...")
            continue
    
    if not all_combined_features:
        print("\n❌ No JSONL files were successfully processed")
        return
    
    # Step 3: Combine all processed features into one DataFrame
    print("\n" + "="*70)
    print("COMBINING ALL PROCESSED DATA")
    print("="*70)
    combined_features_df = pd.concat(all_combined_features, ignore_index=True)
    print(f"Total combined features shape: {combined_features_df.shape}")
    print(f"Combined features date range: {combined_features_df['open_start_time'].min()} to {combined_features_df['open_start_time'].max()}")
    
    # Step 4: Read existing CSV
    print("\n" + "="*70)
    print("READING EXISTING CSV")
    print("="*70)
    print(f"Reading existing CSV file: {csv_file}")
    
    timestamp_cols = ['open_start_time', 'open_end_time', 'close_start_time', 'close_end_time']
    
    try:
        existing_df = pd.read_csv(csv_file)
        print(f"Loaded {len(existing_df)} rows from CSV")
        print(f"CSV columns ({len(existing_df.columns)}): {list(existing_df.columns)[:10]}..." if len(existing_df.columns) > 10 else f"CSV columns: {list(existing_df.columns)}")
        
        # Convert timestamp columns to datetime if they're strings, and ensure they're timezone-naive
        for col in timestamp_cols:
            if col in existing_df.columns:
                if existing_df[col].dtype == 'object':
                    try:
                        existing_df.loc[:, col] = pd.to_datetime(existing_df[col])
                    except:
                        pass  # Keep as is if conversion fails
                # Ensure timezone-naive
                if hasattr(existing_df[col].dtype, 'tz') and existing_df[col].dtype.tz is not None:
                    existing_df.loc[:, col] = pd.to_datetime(existing_df[col]).dt.tz_localize(None)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Step 5: Merge dataframes (removing duplicates)
    print("\n" + "="*70)
    print("MERGING WITH EXISTING CSV")
    print("="*70)
    merged_df = merge_dataframes(combined_features_df, existing_df)
    
    # Step 6: Save merged dataframe
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    print(f"Saving merged data to: {csv_file}")
    merged_df.to_csv(csv_file, index=False)
    print(f"✅ Successfully saved {len(merged_df)} rows to {csv_file}")
    
    print("\n" + "="*70)
    print("SUCCESS! All JSONL events added to CSV without duplicates")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Processed {len(jsonl_files)} JSONL file(s)")
    print(f"  - Added {len(combined_features_df)} total rows from JSONL files")
    print(f"  - Final CSV contains {len(merged_df)} rows")

if __name__ == "__main__":
    main()
