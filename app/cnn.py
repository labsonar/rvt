import torch
import ml.models.cnn as lps_cnn
import ml.trainer as lps_trainer

import lps_rvt.rvt_types as rvt
import lps_rvt.ml_loader as rvt_ml

def main():
    dataloaders = rvt_ml.AudioDataset.get_dataloaders(transform=rvt_ml.SpectrogramTransform())
    train_loader = dataloaders[rvt.Subset.TRAIN]
    val_loader = dataloaders[rvt.Subset.VALIDATION]
    test_loader = dataloaders[rvt.Subset.TEST]

    input_shape = next(iter(train_loader))[0].shape[1:]
    model = lps_cnn.CNN(
        input_shape=input_shape,
        conv_n_neurons=[32, 64],
        conv_activation=torch.nn.ReLU,
        conv_pooling=torch.nn.AvgPool2d,
        conv_pooling_size=[2, 2],
        conv_dropout=0.5,
        batch_norm=torch.nn.BatchNorm2d,
        kernel_size=5,
        classification_n_neurons=[128, 32],
        n_targets=1,
        classification_dropout=0.4,
        classification_norm=torch.nn.BatchNorm1d,
        classification_hidden_activation=torch.nn.ReLU,
        classification_output_activation=torch.nn.Sigmoid
    )

    output_dir = "./results/ml/cnn"

    results_df = lps_trainer.complete_training(
        model=model,
        output_dir=output_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    print(f"Treinamento concluído. Resultados salvos na pasta {output_dir}.")
    print(results_df)

if __name__ == "__main__":
    main()
