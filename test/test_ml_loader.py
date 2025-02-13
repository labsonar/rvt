
import lps_rvt.ml_loader as lps_ml

dataloaders = lps_ml.AudioDataset.get_dataloaders()

for subset, dataloader in dataloaders.items():
    print(subset, ": ", len(dataloader))