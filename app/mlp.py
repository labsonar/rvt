import ml.models.mlp as lps_ml
import ml.trainer as lps_trainer

import lps_rvt.rvt_types as rvt
import lps_rvt.ml_loader as rvt_ml



def main():

    dataloaders = rvt_ml.AudioDataset.get_dataloaders(transform=rvt_ml.SpectrogramTransform())
    train_loader = dataloaders[rvt.Subset.TRAIN]
    val_loader = dataloaders[rvt.Subset.VALIDATION]
    test_loader = dataloaders[rvt.Subset.TEST]

    output_dir = "./results/ml/mlp"

    input_size = next(iter(train_loader))[0].shape[1:]
    model = lps_ml.MLP(input_size, hidden_channels=[128, 32], n_targets=1, dropout=0.2)

    results_df = lps_trainer.complete_training(
            model=model,
            output_dir=output_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader)

    print(f"Treinamento concluído. Resultados salvos na pasta {output_dir}.")
    print(results_df)

    # results_df = lps_trainer.analyze_threshold_variation(
    #         model=model,
    #         output_dir=output_dir,
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         test_loader=test_loader)

    # print(f"Treinamento concluído. Resultados salvos na pasta {output_dir}.")
    # print(results_df)

if __name__ == "__main__":
    main()
