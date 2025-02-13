import pandas as pd
import numpy as np

import lps_rvt.rvt_types as rvt
import lps_rvt.dataloader as lps_loader
import lps_rvt.pipeline as lps_pipeline

loader = lps_loader.DataLoader()
results = pd.read_csv("./data/docs/test_results.csv")
fs=8000
tolerance_before = int(fs*0.001)
tolerance_after = int(fs*0.03)

tns, fps, fns, tps = {}, {}, {}, {}

for ammu in rvt.Ammunition:

    files = loader.get_files(ammunition_types=[ammu])

    tns[ammu], fps[ammu], fns[ammu], tps[ammu] = [], [], [], []

    for file in files:

        expected_detections, expected_rebounds = loader.get_critical_points(file_id=file, fs=fs)

        detect_samples = []
        artifacts_filtered = results[results["Test File ID"] == file]

        for _, artifact in artifacts_filtered.iterrows():
            delta = pd.Timedelta(artifact['Offset']).total_seconds()
            detect_samples.append(int(delta * fs))

        tn, fp, fn, tp = lps_pipeline.Detector.evaluate(
                expected_detections = expected_detections,
                expected_rebounds = expected_rebounds,
                samples_to_check = range(1000),
                detect_samples = detect_samples,
                tolerance_before = tolerance_before,
                tolerance_after = tolerance_after
            ).ravel()

        tns[ammu].append(tn)
        fps[ammu].append(fp)
        fns[ammu].append(fn)
        tps[ammu].append(tp)

        print("\t", file, ": ", tp, "/", fn + tp, "  -> ", fp)


for ammu in rvt.Ammunition:

    tn = np.sum(tns[ammu])
    fp = np.sum(fps[ammu])
    fn = np.sum(fns[ammu])
    tp = np.sum(tps[ammu])

    print(ammu, ": ", tp, "/", fn + tp, "  -> ", fp)
