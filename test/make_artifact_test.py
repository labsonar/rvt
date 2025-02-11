import os
import pandas as pd
import scipy.io.wavfile as scipy_wav
import lps_rvt.dataloader as rvt_loader
import pandas as pd
import datetime

def convert_seconds_to_hms(seconds: float) -> str:
    td = pd.to_timedelta(seconds, unit='s')
    time_str = str(td)
    time_str = time_str.split(" ")[-1]

    if '.' in time_str:
        time_str = time_str.split('.')[0] + '.' + time_str.split('.')[1][:3]
    else:
        time_str = f'{time_str}.000'

    return time_str

def process_and_generate_output(input_csv: str, output_csv: str, output_path: str, interval: int = 60) -> None:
    """
    Process the input CSV to extract audio segments and generate the output CSV with artifact details.
    
    Args:
        input_csv (str): Path to the input CSV file with artifact details.
        output_csv (str): Path to the output CSV file to save processed data.
        output_path (str): Directory path where the output WAV files will be saved.
        interval (int): Interval in seconds.
    """
    artifact_df = pd.read_csv(input_csv)
    loader = rvt_loader.DataLoader()

    os.makedirs(output_path, exist_ok=True)

    output_data = []

    for _, artifact in artifact_df.iterrows():

        if "Tiro" in artifact["Caracterization"] and "Ricochete" not in artifact["Caracterization"]:

            delta = pd.Timedelta(artifact['Offset']).total_seconds()
            fs, data = loader.get_data(artifact["Test File ID"])

            ref_point = int(delta * fs)
            start_sample = ref_point - fs * int(interval/2)
            end_sample = ref_point + fs * int(interval/2)

            start_sample = max(0, start_sample)
            end_sample = min(len(data), end_sample)

            file_id = f'{artifact["Caracterization"].replace(" ","_")}-Boia_{artifact["Bouy"]}'
            filename = os.path.join(output_path, f"{file_id}.wav")

            scipy_wav.write(filename=filename, rate=fs, data=data[start_sample:end_sample])

            for _, other_artifact in artifact_df.iterrows():
                if other_artifact["Test File ID"] == artifact["Test File ID"]:
                    other_delta = pd.Timedelta(other_artifact['Offset']).total_seconds()
                    other_ref_point = int(other_delta * fs)

                    if start_sample <= other_ref_point < end_sample:
                        other_offset = other_ref_point - start_sample
                        relative_offset_in_seconds = other_offset / fs
                        output_data.append({
                            'Test File ID': artifact["Test File ID"],
                            'Artifact File ID': file_id,
                            'Caracterization': other_artifact["Caracterization"],
                            'Bouy': other_artifact["Bouy"],
                            'Offset': convert_seconds_to_hms(relative_offset_in_seconds)
                        })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

def main():
    input_csv = 'data/docs/test_artifacts.csv'
    output_csv = 'data/docs/artifacts_ids.csv'
    output_path = './data/artifacts'

    process_and_generate_output(input_csv, output_csv, output_path)

if __name__ == "__main__":
    main()
