import numpy as np

import ml.visualization as ml
import lps_rvt.rvt_types as rvt
import lps_rvt.ml_loader as lps_ml

dataloaders = lps_ml.AudioDataset.get_dataloaders()

for subset, dataloader in dataloaders.items():
    print(subset, ": ", len(dataloader))


dataset = lps_ml.AudioDataset(subset=rvt.Subset.TRAIN)

feature_list = []
label_list = []

for audio_tensor, label in dataset:
    feature_vector = audio_tensor.numpy()
    feature_list.append(feature_vector)
    label_list.append(label.item())

features, labels = np.array(feature_list), np.array(label_list)

ml.export_tsne(features, labels, "./data/ml/tsne_plot.png")