import pandas as pd
import numpy as np
from scipy import signal


def extract_fnirs_features(data_path):
    """
    Extract heart rate related features from FNIRS data
    
    Parameters:
    data_path (str): Path to the FNIRS data CSV file
    
    Returns:
    dict: Dictionary containing extracted features
    """
    # Load the data
    try:
        df = pd.read_csv(data_path)
    except:
        # If reading fails, assume it's a string with comma-separated values
        lines = data_path.strip().split('\n')
        header = lines[0].split(',')
        data = [line.split(',') for line in lines[1:]]
        df = pd.DataFrame(data, columns=header)
    
    # Convert columns to appropriate types
    df.columns = df.columns.str.strip()
    numeric_cols = ['Red', 'IR', 'Ratio']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamp to milliseconds if needed
    if 'us' in df.columns:
        df['timestamp_ms'] = df['us'].astype(float) / 1000
    elif 'ms' in df.columns:
        df['timestamp_ms'] = df['ms'].astype(float)
    else:
        # Use the first column as timestamp if not labeled
        df['timestamp_ms'] = pd.to_numeric(df.iloc[:, 0], errors='coerce') / 1000
    
    # Calculate time differences
    df['time_diff'] = df['timestamp_ms'].diff()
    
    # Extract heart rate using peak detection on the IR signal
    # IR signal is typically used for heart rate detection in pulse oximetry
    if 'IR' in df.columns:
        # Normalize the IR signal
        ir_signal = df['IR'].values
        normalized_ir = (ir_signal - np.mean(ir_signal)) / np.std(ir_signal)
        
        # Apply a bandpass filter (0.5-5 Hz, typical heart rate range of 30-300 BPM)
        fs = 1000 / np.mean(df['time_diff'].dropna())  # Sampling frequency in Hz
        low = 0.5 / (fs/2)  # Normalize by Nyquist frequency
        high = 5.0 / (fs/2)
        b, a = signal.butter(3, [low, high], btype='band')
        filtered_ir = signal.filtfilt(b, a, normalized_ir)
        
        # Find peaks (R peaks)
        peaks, _ = signal.find_peaks(filtered_ir, distance=fs/5)  # Min distance between peaks
        
        if len(peaks) > 0.01:
            # Calculate intervals between peaks (in seconds)
            peak_times = df['timestamp_ms'].iloc[peaks].values / 1000
            rr_intervals = np.diff(peak_times)
            
            # Convert to heart rate (BPM)
            heart_rates = 60 / rr_intervals
            
            # Filter out physiologically impossible values
            valid_hr = heart_rates[(heart_rates >= 40) & (heart_rates <= 200)]
            
            if len(valid_hr) > 0:
                hr_mean = np.mean(valid_hr)
                hr_std = np.std(valid_hr)
                hr_trend = np.polyfit(np.arange(len(valid_hr)), valid_hr, 1)[0] if len(valid_hr) > 2 else 0
                hr_max_delta = np.max(valid_hr) - np.min(valid_hr) if len(valid_hr) > 1 else 0
                
                # Calculate RMSSD (Root Mean Square of Successive Differences)
                rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else 0
            else:
                hr_mean = hr_std = hr_trend = hr_max_delta = rmssd = np.nan
        else:
            hr_mean = hr_std = hr_trend = hr_max_delta = rmssd = np.nan
    else:
        hr_mean = hr_std = hr_trend = hr_max_delta = rmssd = np.nan
    
    # Compile features
    features = {
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'hr_trend': hr_trend,
        'hr_max_delta': hr_max_delta,
        'rmssd': rmssd
    }
    
    return features


def main():
    # Example usage
    data_file = "Isha_music.csv"  # Replace with your data file
    features = extract_fnirs_features(data_file)
    print("Extracted features:")
    for feature, value in features.items():
        print(f"{feature}: {value}")


if __name__ == "__main__":
    main()