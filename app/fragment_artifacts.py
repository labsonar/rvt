import os
import pandas as pd
import scipy.io as scipy

import lps_rvt.types as rvt
import lps_rvt.dataloader as rvt_loader

artifact_df = pd.read_csv('data/docs/test_artifacts.csv')
output_path = "./data/artifacts"

os.makedirs(output_path, exist_ok=True)

loader = rvt_loader.DataLoader()
files = loader.get_files(ammunition_types=[rvt.Ammunition.EXSUP])

for file_id in files:
    print("### ", file_id)

    df = artifact_df[artifact_df["Test File ID"] == file_id]

    fs, data = loader.get_data(file_id)

    for _, artifact in df.iterrows():
        delta = pd.Timedelta(artifact['Offset']).total_seconds()
        print("\t", artifact["Test File ID"], delta, artifact["Caracterization"], artifact["Bouy"])

        start_sample = int(delta * fs)
        n_samples = int(.25 * fs) #250 milliseconds

        filename = os.path.join(output_path, f'{artifact["Caracterization"]} Boia_{artifact["Bouy"]}.wav')
        scipy.wavfile.write(filename=filename, rate=fs, data=data[start_sample:start_sample+n_samples])