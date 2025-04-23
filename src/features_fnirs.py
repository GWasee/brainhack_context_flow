import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
import sys


def extract_hr_from_fNIRS_with_timestamps(ppg_window, sampling_rate):
    """
    Extract heart rate features from raw fNIRS/PPG signal with timestamps.

    Parameters:
    -----------
    ppg_window : pd.DataFrame
        DataFrame with 'timestamp' and 'ppg_signal' columns
    sampling_rate : int
        Sampling rate of the PPG/fNIRS signal in Hz

    Returns:
    --------
    dict
        Dictionary of extracted HR features (mean, std, trend, etc.)
    """
    if len(ppg_window) < 2:
        return None

    timestamps = ppg_window['timestamp'].values
    signal_values = ppg_window['ppg_signal'].values

    # Bandpass filter for PPG (0.5–4 Hz)
    sos = signal.butter(2, [0.5, 4.0], btype='bandpass', fs=sampling_rate, output='sos')
    filtered = signal.sosfiltfilt(sos, signal_values)

    # Detect peaks (heartbeats)
    min_distance = int(sampling_rate * 0.4)  # 150 BPM max
    peaks, _ = signal.find_peaks(filtered, distance=min_distance, prominence=0.1)

    if len(peaks) < 1:
        print("Not enough peaks detected.")
        return None

    # Convert peak times to BPM
    beat_times = timestamps[peaks]
    rr_intervals = np.diff(beat_times)  # seconds
    bpm_values = 60 / rr_intervals

    # Feature 1: Mean Heart Rate
    hr_mean = np.mean(bpm_values)

    # Feature 2: Heart Rate Standard Deviation
    hr_std = np.std(bpm_values)

    # Feature 3: HR Trend (Linear regression on BPM vs time)
    # time_midpoints = (beat_times[1:] + beat_times[:-1]) / 2
    X = timestamps.reshape(-1, 1)
    y = bpm_values
    model = LinearRegression().fit(X, y)
    hr_trend = model.coef_[0]

    # Feature 4: Max HR Delta
    hr_diff = np.abs(np.diff(bpm_values))
    hr_max_delta = np.max(hr_diff) if len(hr_diff) > 0 else 0

    # Feature 5: RMSSD (HRV)
    rr_ms = 60000 / bpm_values
    rr_diff = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(rr_diff**2)) if len(rr_diff) > 0 else 0

    features = {
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'hr_trend': hr_trend,
        'hr_max_delta': hr_max_delta,
        'rmssd': rmssd
    }

    return features


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_hr_from_fNIRS_csv.py <path_to_csv> [sampling_rate]")
        sys.exit(1)

    csv_file = sys.argv[1]
    sampling_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 50  # default 50Hz

    try:
        df = pd.read_csv(csv_file)

        # Auto-adjust columns: 'us' → 'timestamp' and 'Ratio' → 'ppg_signal'
        if {'us', 'Ratio'}.issubset(df.columns):
            df = df.rename(columns={'us': 'timestamp', 'Ratio': 'ppg_signal'})
            df['timestamp'] = df['timestamp'] / 1e6  # convert from microseconds to seconds
        else:
            raise ValueError("CSV must contain 'us' and 'Ratio' columns.")

        # Run HR extraction
        features = extract_hr_from_fNIRS_with_timestamps(df[['timestamp', 'ppg_signal']], sampling_rate)

        if features:
            print("Extracted HR Features:")
            for key, value in features.items():
                print(f"  {key}: {value:.2f}")

    except Exception as e:
        print("Error processing file:", e)
