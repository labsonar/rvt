"""Script to calculate the correlation between two audios signals."""
import argparse
import datetime
import typing
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import os

import src.rvt.test_loader as test_loader
import src.rvt.loader as loader
import src.rvt.artifact as artifact
from lps_sp.signal import decimate
            

def load_audio(artifact_id: int, buoy_id: int, signal_type: str):
    """Loads the audio data.
    
    Args:
        artifact_id (int): The ID of the audio from which to get the data.
        buoy_id (int): The buoy related to the audio.
        signal_type (str): The type of the audio signal, artifact or background.
    
    Returns:
        fs (int): sampling rate.
        audio: audio data.
    """

    loader_ = loader.DataLoader()
    manager_ = artifact.ArtifactManager()
    time = manager_.get_time(artifact_id, buoy_id)

    if signal_type == 'artifact':
        start = time - datetime.timedelta(seconds=0.05)
        end = time + datetime.timedelta(seconds=0.25)
    elif signal_type == 'background':
        start = time - datetime.timedelta(seconds=10)
        end = time - datetime.timedelta(seconds=2)

    fs, audio = loader_.get_data(buoy_id, start, end)
    return fs, audio


def correlation(audio_1: np.ndarray, audio_2: np.ndarray) -> np.ndarray:
    """Claculates the correlation between two audio signals.

    Args:
        audio_1 (np.ndarray): One of the audios to be included in the correlation.
        audio_2 (np.ndarray): The other one.

    Returns:
        correlation_ (np.ndarray): The correlation array.
    """

    # normalization
    audio_1 = audio_1 - np.mean(audio_1)
    audio_1 = audio_1 / np.max(np.abs(audio_1))
       
    audio_2 = audio_2 - np.mean(audio_2)
    audio_2 = audio_2 / np.max(np.abs(audio_2))
    
    correlation_ = scipy.signal.correlate(audio_1, audio_2, mode='full')
    return correlation_


def plot(args: argparse.Namespace, fs_1: int, audio_1: np.ndarray, 
         fs_2: int, audio_2: np.ndarray, output_filename: str):
    """Plots the correlation between two audio signals.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        fs_1 (int): Sampling rate of the first audio.
        audio_1 (np.ndarray): Audio data.
        fs_2 (int): Sampling rate of the second audio.
        audio_2 (np.ndarray): Audio data.
        output_filename (str): The name of the image file being generated.
    """ 
    
    if fs_1 != fs_2:
        print('The audios sampling rates are different.')
        if fs_1 > fs_2:
            audio_1 = decimate(audio_1, fs_1/fs_2)
        else:
            audio_2 = decimate(audio_2, fs_2/fs_1)
    
    correlation_ = correlation(audio_1, audio_2)
    x_axis = scipy.signal.correlation_lags(len(audio_1), len(audio_2))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, correlation_)
    plt.title(f'Correlation {output_filename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation')
    plt.grid()
    path = f'data/Correlation'
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, output_filename))
    plt.close()
    print(f"Audio Correlation saved as {output_filename}")


def my_tuple (tuple_str: str) -> typing.Tuple[int, int, str]:
    """Converts an 'a,b,c' formatted string to a (a, b, c) tuple."""

    try:
        split = tuple_str.split(',')

        if len(split) != 3:
            raise ValueError(f"Expected 3 elements, but got {len(split)}.")
        if split[2] not in ('artifact', 'background'):
            raise ValueError(f'Signal type \"{split[2]}\" is not allowed. Try \"artifact\" or \"background\".')
        
        return (int(split[0]), int(split[1]), split[2])
    except ValueError:
        return argparse.ArgumentTypeError(f'Invalid tuple: {tuple_str}. Try  the \'a,b,c\' format.')


def main():
    """Main fuction to parse command line arguments."""

    parser = argparse.ArgumentParser(description='Audio correlation.')
    parser.add_argument('-f', '--first', type=my_tuple, required=True, default=None, 
                        help='Identification of the first audio to be included in the correlation \
                        (e.g.: \'artifact_ID,buoy_ID,signal_type\'). Signal types: \"artifact\" or \"background\".')
    parser.add_argument('-s', '--second', type=my_tuple, default=None, 
                        help='Identification of the second audio to be included in the correlation (optional) \
                        (e.g.: \'artifact_ID,buoy_ID,signal_type\'). Signal types: \"artifact\" or \"background\".')
    parser.add_argument('-tf', '--training_files', action='store_true', 
                       help='If selected, the script plots the correlation between a chosen audio file \
                        and each one of the training files.')
    
    args = parser.parse_args()

    if not args.training_files and args.second is None:
        parser.error('The \"--second\" argument needs to be included if \"--training_files\" is not selected.')

    if args.training_files:
        loader_ = test_loader.DataLoader()
        for num in range(1, 27):
            output_filename = f'{args.first[0]}_{args.first[1]}_{args.first[2]}_AND_test_{num}'
            fs_1, audio_1 = load_audio(args.first[0], args.first[1], args.first[2])
            fs_2, audio_2 = loader_.getData(file_ID=num)
            plot(args, fs_1, audio_1, fs_2, audio_2, output_filename)
    else:
        output_filename = f'{args.first[0]}_{args.first[1]}_{args.first[2]}_AND_\
{args.second[0]}_{args.second[1]}_{args.second[2]}'
        fs_1, audio_1 = load_audio(args.first[0], args.first[1], args.first[2])
        fs_2, audio_2 = load_audio(args.second[0], args.second[1], args.second[2])
        plot(args, fs_1, audio_1, fs_2, audio_2, output_filename)


if __name__ == "__main__":
    main()