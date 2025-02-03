""" Script to process test artifacts and generates audio artifacts """
import os
import argparse
import pandas as pd
import scipy.io.wavfile as scipy_wav

import lps_rvt.types as rvt
import lps_rvt.dataloader as rvt_loader

def process_files(output_path: str, ammunition_types: list) -> None:
    """
    Processes the test artifacts to extract and save corresponding signal segments as WAV files.

    Args:
        output_path (str): Directory path where the output WAV files will be saved.
        ammunition_types (list): List of ammunition types to filter the files by.
    """
    artifact_df = pd.read_csv('data/docs/test_artifacts.csv')
    loader = rvt_loader.DataLoader()

    os.makedirs(output_path, exist_ok=True)

    files = loader.get_files(ammunition_types=ammunition_types)

    for file_id in files:
        print("### Processing file:", file_id)

        df = artifact_df[artifact_df["Test File ID"] == file_id]
        fs, data = loader.get_data(file_id)

        for _, artifact in df.iterrows():
            delta = pd.Timedelta(artifact['Offset']).total_seconds()
            print("\tArtifact details: File ID:", artifact["Test File ID"], "Offset:", delta,
                  "Characterization:", artifact["Caracterization"], "Buoy:", artifact["Bouy"])

            start_sample = int(delta * fs)
            n_samples = int(0.25 * fs)  # 250 milliseconds

            filename = os.path.join(output_path,
                                f'{artifact["Caracterization"]} Boia_{artifact["Bouy"]}.wav')
            scipy_wav.write(filename=filename, rate=fs,
                                data=data[start_sample:start_sample + n_samples])

def main() -> None:
    """ Main function os script """

    parser = argparse.ArgumentParser(
        description="Test DataLoader with filters and processing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-a", "--ammunition", nargs="*",
                             choices=[e.name for e in rvt.Ammunition],
                             default=[rvt.Ammunition.EXSUP.name],
                             help="Filter by ammunition types.")

    parser.add_argument("-o", "--output-dir", type=str, default="./data/artifacts",
                        help="Directory to save the output WAV files.")

    args = parser.parse_args()

    ammunition_types = [rvt.Ammunition[t] for t in args.ammunition] if args.ammunition else None

    if len(ammunition_types) == 0:
        print("No ammunition type selected")
        return

    process_files(args.output_dir, ammunition_types=ammunition_types)

if __name__ == "__main__":
    main()
