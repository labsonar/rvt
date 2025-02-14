import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile

import lps_rvt.rvt_types
import lps_rvt.dataloader

def main():
    parser = argparse.ArgumentParser(description="Split artifact and background for ML training")
    parser.add_argument("--sample_size", type=float, default=0.25, help="Duration of each sample in seconds")
    parser.add_argument("--outputdir", type=str, default="./data/ml", help="Output directory")
    args = parser.parse_args()

    loader = lps_rvt.dataloader.DataLoader()

    sample_size = args.sample_size
    outputdir = args.outputdir
    os.makedirs(outputdir, exist_ok=True)

    samples_data = []

    for subset in lps_rvt.rvt_types.Subset:
        detection_dir = os.path.join(outputdir, "detection")
        bg_dir = os.path.join(outputdir, "bg")

        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(bg_dir, exist_ok=True)

        selected_files = loader.get_files(subsets=[subset])

        for file in selected_files:
            fs, data = loader.get_data(file)
            expected_detections, expected_rebounds = loader.get_critical_points(file, fs)

            n_samples = int(fs * sample_size)

            for i, start in enumerate(expected_detections):
                if start + n_samples <= len(data):
                    clip = data[start:start + n_samples]
                    output_file = os.path.join(detection_dir, f"{file}_det_{i}.wav")
                    wavfile.write(output_file, fs, clip.astype(np.int16))

                    samples_data.append({
                        'outputfile': output_file,
                        'file': file,
                        'Subset': str(subset),
                        'Classification': 1  # 1 para deteções
                    })

            all_critical_points = set(expected_detections) | set(expected_rebounds)
            min_distance = 10 * fs # 10segundo

            bg_samples = []
            last_included = 0
            for start in range(0, len(data) - n_samples, n_samples):
                if all(abs(start - pt) >= min_distance for pt in all_critical_points) and (start - last_included >= 120*fs):
                    bg_samples.append(start)
                    last_included = start

            for i, start in enumerate(bg_samples):
                clip = data[start:start + n_samples]
                output_file = os.path.join(bg_dir, f"{file}_bg_{i}.wav")
                wavfile.write(output_file, fs, clip.astype(np.int16))

                samples_data.append({
                    'outputfile': output_file,
                    'file': file,
                    'Subset': str(subset),
                    'Classification': 0  # 0 para fundo
                })

    df = pd.DataFrame(samples_data)

    csv_file = os.path.join("./data/docs", "ml_info.csv")
    df.to_csv(csv_file, index=False)

    print(f"Data saved to {csv_file}")

if __name__ == "__main__":
    main()
