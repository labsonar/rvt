"""Script to test the Audio Analysis module."""
import argparse
import datetime
import os
import shutil
import typing

import src.rvt.artifact as artifact
import src.rvt.loader as loader
import src.rvt.analysis as analysis


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


def generate_plots(loader_: loader.DataLoader, artifact_id: int, buoy_id: int, path: str, time: datetime,
                   title: any, plot_types: typing.List[str], signal_type: str = None, 
                   compare: bool = False) -> None:
    """
    Generates and saves various analysis plots (Waveform, PSD, FFT, LOFAR) for a given audio signal.
    
    This function processes artifact and background audio data, and generates the selected plots
    for analysis. If comparison is enabled, it compares the artifact with the background signal.

    Args:
        loader_ (DataLoader): Instance for loading audio data.
        artifact_id (int): ID of the artifact related to the data.
        buoy_id (int): ID of the buoy that recorded the data.
        path (str): Directory path to save generated plots.
        time (datetime): Timestamp of the artifact.
        title (any): Title or identifier for the plot files.
        plot_types (list): List of plot types to generate (e.g., 'Waveform', 'PSD', 'FFT', 'LOFAR').
        compare (bool, optional): Flag to enable comparison between artifact and background signals. Default is False.
    """

    fs, audio = load_audio(artifact_id, buoy_id, 'artifact')
    bkg_fs, bkg_audio = load_audio(artifact_id, buoy_id, 'background')
    
    if compare:
        std_path = os.path.join(path, 'Comparison')
    elif signal_type == 'artifact':
        std_path = os.path.join(path, 'Artifact')
    elif signal_type == 'background':
        std_path = os.path.join(path, 'Background')
        
    analysis_ = analysis.AudioAnalysis(std_path, time)
    
    if 'Waveform' in plot_types and not compare:
        analysis_.waveform(audio, fs, f'{time.strftime('%H_%M_%S')}_{title}.png')
    
    elif 'PSD' in plot_types:
        if compare:
            psd_freq, psd_result = analysis_.psd(audio, fs)
            bkg_psd_freq, bkg_psd_result = analysis_.psd(bkg_audio, bkg_fs)
            analysis_.plot(psd_freq, psd_result, 'PSD', 'Frequency [Hz]', 'Power [W/Hz]', 
                       f'{time.strftime('%H_%M_%S')}_PSD_{title}.png',
                        bkg_psd_freq, bkg_psd_result, compare)
        elif signal_type == 'artifact':
            psd_freq, psd_result = analysis_.psd(audio, fs)
            analysis_.plot(psd_freq, psd_result, 'PSD', 'Frequency [Hz]', 'Power [W/Hz]', 
                       f'{time.strftime('%H_%M_%S')}_PSD_{title}.png')
        elif signal_type == 'background':
            bkg_psd_freq, bkg_psd_result = analysis_.psd(bkg_audio, bkg_fs)
            analysis_.plot(bkg_psd_freq, bkg_psd_result, 'PSD', 'Frequency [Hz]', 'Power [W/Hz]', 
                       f'{time.strftime('%H_%M_%S')}_PSD_{title}.png')
    
    elif 'FFT' in plot_types:
        if compare:
            fft_freq, fft_result = analysis_.fft(audio, fs)
            bkg_fft_freq, bkg_fft_result = analysis_.fft(bkg_audio, bkg_fs)
            analysis_.plot(fft_freq, fft_result, 'FFT', 'Frequency [Hz]', 'Magnitude [dB]', 
                           f'{time.strftime('%H_%M_%S')}_FFT_{title}.png', 
                           bkg_fft_freq, bkg_fft_result, compare)
        elif signal_type == 'artifact':
            fft_freq, fft_result = analysis_.fft(audio, fs)
            analysis_.plot(fft_freq, fft_result, 'FFT', 'Frequency [Hz]', 'Magnitude [dB]', 
                           f'{time.strftime('%H_%M_%S')}_FFT_{title}.png')
        elif signal_type == 'background':
            bkg_fft_freq, bkg_fft_result = analysis_.fft(bkg_audio, bkg_fs)
            analysis_.plot(bkg_fft_freq, bkg_fft_result, 'FFT', 'Frequency [Hz]', 'Magnitude [dB]', 
                           f'{time.strftime('%H_%M_%S')}_FFT_{title}.png')
    
    if 'LOFAR' in plot_types and not compare:
        analysis_.lofar(audio, fs, f'{time.strftime('%H_%M_%S')}_LOFAR_{title}.png')


# TODO pylint ta reclamando do numero de variaveis da funcao # pylint: disable=fixme
def manage_plots(loader_: loader.DataLoader, manager: artifact.ArtifactManager,
                 args: argparse.Namespace) -> None:
    """
    Manages the generation of plots for different artifact types and signal types.
    """

    artifact_ids = []
    for artifact_id in args.artifact_ids:
        if "-" in artifact_id:
            start, end = map(int, artifact_id.split("-"))
            artifact_ids.extend(range(start, end + 1))
        else:
            artifact_ids.append(int(artifact_id))

    for artifact_type in args.artifact_types:
        for artifact_id in artifact_ids:
            if artifact_id in manager.id_from_type(artifact_type):
                for buoy_id in manager[artifact_id]:

                    time = manager.get_time(artifact_id, buoy_id)

                    path = f'data/Analysis/{artifact_type}/Boia{buoy_id}'

                    if 'artifact' in args.signal_type and not args.compare:
                        generate_plots(loader_, artifact_id, buoy_id, path=path, time=time, 
                                       title=artifact_id, plot_types=args.plot_types, 
                                       signal_type=args.signal_type)
                        
                    if 'background' in args.signal_type and not args.compare:
                        generate_plots(loader_, artifact_id, buoy_id, path=path, time=time, 
                                       title=artifact_id, plot_types=args.plot_types, 
                                       signal_type=args.signal_type)
                        
                    if args.compare:
                        generate_plots(loader_, artifact_id, buoy_id, path, time, artifact_id, 
                                       plot_types=args.plot_types, compare=args.compare)
            else:
                print(f"Artifact {artifact_id} not from chosen artifact type {artifact_type}")
                print(f"Possible {artifact_type} artifacts: {manager.id_from_type(artifact_type)}")
        
        
def main():
    """
    Main function to parse command-line arguments and initiate the audio analysis process.

    This function sets up argument parsing for artifact analysis, initializes data loading,
    and triggers the generation of analysis plots based on user input.
    """
    parser = argparse.ArgumentParser(description="Audio analysis and plot")
    parser.add_argument("-id", "--artifact_ids", type=str, nargs="+", required=True,
                        help="List or range of artifact IDs (e.g., 1 2 3 or 1-5).")
    parser.add_argument("-t", "--artifact_types", nargs='+', default=['EX-SUP', 'GAE', 'HE3m'],
                        type=str, help='Lista de tipos de municao')
    parser.add_argument("-p", "--plot_types", nargs='+', default=['Waveform', 'PSD', 'FFT',
                                                                  'LOFAR'],
                        type=str, help='Lista de tipos de plot')
    parser.add_argument("-s", "--signal_type",  default=[], type=str, choices=["artifact", "background"],
                        help="Signal type to analyze (artifact or background).")
    parser.add_argument("-c", "--compare", action="store_true",
                        help="Compare artifact with background.")
    
    args = parser.parse_args()

    if os.path.exists("data/Analysis"):
        shutil.rmtree("data/Analysis") # Usando so para testes
    os.makedirs("data/Analysis")

    manager_ = artifact.ArtifactManager()
    loader_ = loader.DataLoader()

    manage_plots(loader_, manager_, args)

if __name__ == "__main__":

    main()
