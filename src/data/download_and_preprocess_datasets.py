import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import scipy.io

def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {output_path}...")
    try:
        # User-Agent header to avoid bot blockers (especially for figshare/github raw)
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise e

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

def segment_sliding_window(X, y, window_size, overlap_ratio=0.5):
    step = int(window_size * (1.0 - overlap_ratio))
    X_segmented = []
    y_segmented = []
    
    for start in range(0, len(X) - window_size + 1, step):
        end = start + window_size
        window_y = y[start:end]
        majority_label = np.argmax(np.bincount(window_y.astype(int)))
        X_segmented.append(X[start:end].T)  # Shape (C, window_size)
        y_segmented.append(majority_label)
        
    if len(X_segmented) == 0:
        return np.empty((0, X.shape[1], window_size)), np.empty((0,))
    return np.stack(X_segmented), np.array(y_segmented)

def normalize_features(X):
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std == 0.0] = 1.0  # Avoid division by zero
    return (X - mean) / std

# =====================================================================
# 1. MotionSense Preprocessing
# =====================================================================
def preprocess_motionsense(data_dir):
    cache_path = os.path.join(data_dir, "motion_sense", "motionsense_preprocessed.npz")
    if os.path.exists(cache_path):
        print(f"MotionSense preprocessed cache already exists: {cache_path}")
        return
        
    motionsense_dir = os.path.join(data_dir, "motion_sense")
    zip_path = os.path.join(motionsense_dir, "A_DeviceMotion_data.zip")
    os.makedirs(motionsense_dir, exist_ok=True)
    
    url = "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip"
    download_file(url, zip_path)
    extract_zip(zip_path, motionsense_dir)
    
    extract_path = os.path.join(motionsense_dir, "A_DeviceMotion_data")
    if not os.path.exists(extract_path):
        # Check if extracted directly to motionsense_dir
        extract_path = motionsense_dir
        
    print("Preprocessing MotionSense dataset...")
    activity_mapping = {'dws': 0, 'ups': 1, 'wlk': 2, 'jog': 3, 'sit': 4, 'std': 5}
    
    X_list = []
    y_list = []
    
    # Iterate over directories starting with prefixes
    for folder in sorted(os.listdir(extract_path)):
        folder_path = os.path.join(extract_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Identify class from folder prefix
        prefix = folder.split('_')[0]
        if prefix not in activity_mapping:
            continue
        label_val = activity_mapping[prefix]
        
        # Process each subject file (sub_1.csv to sub_24.csv)
        for filename in sorted(os.listdir(folder_path)):
            if not filename.endswith('.csv'):
                continue
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path).dropna()
            
            # Extract 12 columns
            cols = [
                'attitude.roll', 'attitude.pitch', 'attitude.yaw',
                'gravity.x', 'gravity.y', 'gravity.z',
                'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
                'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'
            ]
            sig = df[cols].values
            
            y_continuous = np.full(len(sig), label_val)
            X_seg, y_seg = segment_sliding_window(sig, y_continuous, window_size=400, overlap_ratio=0.5)
            
            if len(X_seg) > 0:
                X_list.append(X_seg)
                y_list.append(y_seg)
                
    if len(X_list) == 0:
        raise ValueError("MotionSense preprocessing resulted in empty dataset.")
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Global channel-wise normalization to preserve relative amplitude/intensity features
    print("Applying global channel-wise normalization to MotionSense...")
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    std[std == 0.0] = 1.0
    X = (X - mean) / std
    
    print(f"Saving preprocessed MotionSense to {cache_path}...")
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"MotionSense Preprocessing Complete! Shape: X={X.shape}, y={y.shape}")

# =====================================================================
# 2. USC-HAD Preprocessing
# =====================================================================
def preprocess_uschad(data_dir):
    cache_path = os.path.join(data_dir, "USC-HAD", "uschad_preprocessed.npz")
    if os.path.exists(cache_path):
        print(f"USC-HAD preprocessed cache already exists: {cache_path}")
        return
        
    uschad_dir = os.path.join(data_dir, "USC-HAD")
    zip_path = os.path.join(uschad_dir, "usc_had.zip")
    os.makedirs(uschad_dir, exist_ok=True)
    
    url = "https://sipi.usc.edu/had/USC-HAD.zip"
    download_file(url, zip_path)
    extract_zip(zip_path, uschad_dir)
    
    print("Preprocessing USC-HAD dataset...")
    X_list = []
    y_list = []
    
    # USC-HAD folder structure has Subject1 to Subject14
    for subj_idx in range(1, 15):
        subj_dir = os.path.join(uschad_dir, f"Subject{subj_idx}")
        if not os.path.exists(subj_dir):
            subj_dir = os.path.join(uschad_dir, "USC-HAD", f"Subject{subj_idx}")
            if not os.path.exists(subj_dir):
                # Try lowercase
                subj_dir = os.path.join(uschad_dir, f"subject{subj_idx}")
                if not os.path.exists(subj_dir):
                    subj_dir = os.path.join(uschad_dir, "USC-HAD", f"subject{subj_idx}")
                    if not os.path.exists(subj_dir):
                        continue
                
        for filename in sorted(os.listdir(subj_dir)):
            if not filename.endswith('.mat'):
                continue
            file_path = os.path.join(subj_dir, filename)
            
            # Filename format: aXtY.mat where X is activity (1-12)
            try:
                # E.g. 'a1t1.mat'
                act_str = filename.split('a')[1].split('t')[0]
                label_val = int(act_str) - 1  # 0-indexed
            except:
                continue
                
            mat = scipy.io.loadmat(file_path)
            if 'sensor_readings' not in mat:
                continue
            sig = mat['sensor_readings'].astype(np.float32)
            
            y_continuous = np.full(len(sig), label_val)
            X_seg, y_seg = segment_sliding_window(sig, y_continuous, window_size=250, overlap_ratio=0.5)
            
            if len(X_seg) > 0:
                X_list.append(X_seg)
                y_list.append(y_seg)
                
    if len(X_list) == 0:
        raise ValueError("USC-HAD preprocessing resulted in empty dataset.")
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Global channel-wise normalization to preserve relative amplitude/intensity features
    print("Applying global channel-wise normalization to USC-HAD...")
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    std[std == 0.0] = 1.0
    X = (X - mean) / std
    
    print(f"Saving preprocessed USC-HAD to {cache_path}...")
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"USC-HAD Preprocessing Complete! Shape: X={X.shape}, y={y.shape}")

# =====================================================================
# 3. MobiAct Preprocessing
# =====================================================================
def parse_mobiact_txt(filepath):
    """Parses timestamps and values from MobiAct format text files."""
    timestamps = []
    values = []
    
    start_data = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('@DATA'):
                start_data = True
                continue
            if start_data:
                parts = line.split(',')
                if len(parts) >= 4:
                    timestamps.append(int(parts[0]))
                    values.append([float(p) for p in parts[1:4]])
                    
    return np.array(timestamps), np.array(values)

def preprocess_mobiact(data_dir):
    cache_path = os.path.join(data_dir, "MobiAct", "mobiact_preprocessed.npz")
    if os.path.exists(cache_path):
        print(f"MobiAct preprocessed cache already exists: {cache_path}")
        return
        
    mobiact_dir = os.path.join(data_dir, "MobiAct")
    if not os.path.exists(mobiact_dir):
        raise FileNotFoundError(f"MobiAct directory not found at {mobiact_dir}. Please clone it first.")
        
    print("Preprocessing MobiAct dataset...")
    # Activity mapping for MobiAct/MobiFall v2.0 (9 ADL classes only, excluding falls)
    adl_fall_mapping = {
        'STD': 0, 'WAL': 1, 'JOG': 2, 'JUM': 3, 'STU': 4,
        'STN': 5, 'SCH': 6, 'CSI': 7, 'CSO': 8
    }
    
    # Group files by (ACT_CODE, SUB_ID, TRIAL_NO)
    file_groups = {}
    
    # Traverse subdirectories
    for root, dirs, files in os.walk(mobiact_dir):
        for filename in files:
            if not filename.endswith('.txt') or filename == 'DataDescribe.txt':
                continue
                
            # Name format: ACT_SENSOR_SUB_TRIAL.txt (e.g. STU_acc_11_4.txt)
            parts = filename[:-4].split('_')
            if len(parts) != 4:
                continue
            act, sensor, sub, trial = parts
            
            if act not in adl_fall_mapping:
                continue
                
            group_key = (act, sub, trial)
            if group_key not in file_groups:
                file_groups[group_key] = {}
            file_groups[group_key][sensor] = os.path.join(root, filename)
            
    print(f"Found {len(file_groups)} trials to process.")
    X_list = []
    y_list = []
    
    # 50 Hz grid step in ns (20 ms = 20,000,000 ns)
    step_ns = 20_000_000
    
    for idx, (group_key, sensors) in enumerate(sorted(file_groups.items())):
        if 'acc' not in sensors or 'gyro' not in sensors:
            continue
            
        act, sub, trial = group_key
        label_val = adl_fall_mapping[act]
        
        try:
            t_acc, val_acc = parse_mobiact_txt(sensors['acc'])
            t_gyro, val_gyro = parse_mobiact_txt(sensors['gyro'])
            
            if len(t_acc) < 10 or len(t_gyro) < 10:
                continue
                
            t_start = max(t_acc.min(), t_gyro.min())
            t_end = min(t_acc.max(), t_gyro.max())
            
            if t_start >= t_end:
                continue
                
            # 50 Hz grid interpolation
            t_grid = np.arange(t_start, t_end, step_ns)
            if len(t_grid) < 128:
                continue
                
            acc_x = np.interp(t_grid, t_acc, val_acc[:, 0])
            acc_y = np.interp(t_grid, t_acc, val_acc[:, 1])
            acc_z = np.interp(t_grid, t_acc, val_acc[:, 2])
            
            gyro_x = np.interp(t_grid, t_gyro, val_gyro[:, 0])
            gyro_y = np.interp(t_grid, t_gyro, val_gyro[:, 1])
            gyro_z = np.interp(t_grid, t_gyro, val_gyro[:, 2])
            
            sig = np.column_stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
            
            y_continuous = np.full(len(sig), label_val)
            X_seg, y_seg = segment_sliding_window(sig, y_continuous, window_size=128, overlap_ratio=0.5)
            
            if len(X_seg) > 0:
                X_list.append(X_seg)
                y_list.append(y_seg)
        except Exception as e:
            print(f"Error parsing trial {group_key}: {e}")
            
        if (idx + 1) % 200 == 0 or (idx + 1) == len(file_groups):
            print(f"  Processed {idx+1}/{len(file_groups)} trials...")
            
    if len(X_list) == 0:
        raise ValueError("MobiAct preprocessing resulted in empty dataset.")
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Global channel-wise normalization to preserve relative amplitude/intensity features
    print("Applying global channel-wise normalization to MobiAct...")
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    std[std == 0.0] = 1.0
    X = (X - mean) / std
    
    print(f"Saving preprocessed MobiAct to {cache_path}...")
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"MobiAct Preprocessing Complete! Shape: X={X.shape}, y={y.shape}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data")
    
    print("=== HAR Datasets Download & Preprocessing Pipeline ===")
    
    # 1. MotionSense
    try:
        preprocess_motionsense(data_dir)
    except Exception as e:
        print(f"Failed preprocessing MotionSense: {e}")
        
    # 2. USC-HAD
    try:
        preprocess_uschad(data_dir)
    except Exception as e:
        print(f"Failed preprocessing USC-HAD: {e}")
        
    # 3. MobiAct
    try:
        preprocess_mobiact(data_dir)
    except Exception as e:
        print(f"Failed preprocessing MobiAct: {e}")
        
    print("\nPre-processing Pipeline finished!")
