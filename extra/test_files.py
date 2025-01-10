import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import os

INPUT_DIR = "../data/raw_data/"
OUTPUT_DIR = "../data/test_files"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("./test_files.csv", delimiter=",")

for file_id in df['Test File'].unique():
    print(f"Processing Test File: {file_id}")

    frag_df = df[df["Test File"] == file_id]

    audio_data = []
    sample_rate = None

    for i in range(len(frag_df)):
        day = frag_df.iloc[i]['Day']
        bouy = frag_df.iloc[i]['Bouy']
        raw_file = frag_df.iloc[i]['Raw File']
        file_path = os.path.join(INPUT_DIR, f"{day}", f"boia{bouy}", raw_file)

        print(f"\tReading {file_path}")

        try:
            sr, data = wavfile.read(file_path)

            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                print(f"\tWarning: Sample rate mismatch in {file_path}")

            audio_data.append(data)
        except FileNotFoundError:
            print(f"\tWarning: File {file_path} not found!")

    # Se houver dados de áudio, faz o processamento
    if audio_data:
        combined_audio = np.concatenate(audio_data)

        print(sample_rate)
        if sample_rate != 8000:
            print(f"\tResampling {file_id} to 8 kHz")

            num_samples = int(len(combined_audio) * 8000 / sample_rate)
            resampled_audio = resample(combined_audio, num_samples)

            output_file_resampled = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
            wavfile.write(output_file_resampled, 8000, resampled_audio.astype(combined_audio.dtype))
            print(f"\tExported: {output_file_resampled} (8 kHz)")

            output_file = os.path.join(OUTPUT_DIR, f"{file_id}-{sample_rate}.wav")
            wavfile.write(output_file, sample_rate, combined_audio)
            print(f"\tExported: {output_file} with {len(combined_audio) / sample_rate / 60:.2f} minutes")

        else:
            output_file = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
            wavfile.write(output_file, sample_rate, combined_audio)
            print(f"\tExported: {output_file} with {len(combined_audio) / sample_rate / 60:.2f} minutes")
