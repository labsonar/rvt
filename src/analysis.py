import wave
import os
import shutil
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# import lps_sp.signal as lps_signal
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.signal as lps_signal
from artifact import ArtifactManager
from loader import DataLoader

class AudioAnalysis:
    """ Class for audio analysis operations.

    This class provides methods for plotting waveforms,
    computing Power Spectral Density (PSD), 
    performing Fast Fourier Transform (FFT),
    and generating LOFAR analysis plots from an audio signal.
    
    Attributes:
        audio (ndarray): The audio data loaded for analysis.
        fs (int): The sampling frequency of the audio data.
        duration (float): Duration of the audio signal.
        n_samples (int): Number of samples in the audio signal.
        data_path (str): Path to save analysis results.
        time (datetime): Time associated with the audio file for reference.
    """

    # TODO Revisar se todos esses inputs sao realmente necessarios, pylint ta reclamando # pylint: disable=fixme
    def __init__(self, audio, fs, duration, n_samples, data_path, time):
        self.audio_file = time
        self.audio = audio
        self.fs = fs
        self.duration = duration
        self.n_samples = n_samples
        self.time_axis = None
        self.data_path = data_path

    def save(self, output_filename: str) -> None:
        """Save the audio in .wav format

        Args:
            output_filename (str): Name of saved file.
        """

        os.makedirs(self.data_path,exist_ok=True)
        path = os.path.join(self.data_path,output_filename)
        if self.audio is not None :
            scipy.io.wavfile.write(path, self.fs, self.audio)
            print(f"Audio saved as {output_filename}")
        else :
            print("Nenhum áudio foi carregado ou extraído.")

    def plot(self, output_filename):
        """ Generates and saves the waveform plot of the audio.

        Args:
            output_filename (str): The filename for saving the plot.
        """

        self.time_axis = np.linspace(0, self.duration, num=self.n_samples)

        if self.audio is not None and self.time_axis is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(self.time_axis, self.audio)
            plt.title(f"Audio Waveform - {self.audio_file}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.grid()
            os.makedirs(self.data_path, exist_ok=True)
            plt.savefig(f"{self.data_path}/{output_filename}")
            plt.close()
            print(f"Audio Waveform saved as {output_filename}")
        else:
            print("Nenhum áudio foi carregado ou extraído.")

    def psd(self, output_filename):
        """ Generates and saves the Power Spectral Density (PSD) plot.

        Args:
            output_filename (str): The filename for saving the plot.
        """
        
        psd_freq, psd_result = lps_bb.psd(signal=self.audio, fs=self.fs, window_size=4096, overlap=0.5)

        plt.figure(figsize=(12, 6))
        plt.plot(psd_freq, psd_result)
        plt.title(f"PSD - {self.audio_file}")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power [W/Hz]')
        plt.grid()
        os.makedirs(self.data_path, exist_ok=True)
        plt.savefig(f"{self.data_path}/{output_filename}")
        plt.close()
        print(f"PSD saved as {output_filename}")

    def fft(self, output_filename):
        """ Generates and saves the Fast Fourier Transform (FFT) plot.

        Args:
            output_filename (str): The filename for saving the plot.
        """

        fft_result = np.fft.fft(self.audio)
        fft_freq = np.fft.fftfreq(len(self.audio), 1/self.fs)

        magnitude = np.abs(fft_result)[:len(fft_result)//2]
        magnitude[magnitude == 0] = 1e-10

        plt.figure()
        plt.plot(fft_freq[:len(fft_result)//2], 
                 20*np.log10(magnitude))
        # plt.plot(fft_freq, np.abs(fft_result))
        plt.title(f"FFT - {self.audio_file}")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid()
        os.makedirs(self.data_path, exist_ok=True)
        plt.savefig(f"{self.data_path}/{output_filename}")
        plt.close()
        print(f"FFT saved as {output_filename}")

    def lofar(self, output_filename):
        """ Generates and saves LOFAR analysis plot.

        Args:
            output_filename (str): The filename for saving the plot.
        """

        power, freq, time = lps_analysis.SpectralAnalysis.lofar(self.audio, self.fs)

        plt.figure()
        plt.pcolormesh(time, freq, power, shading='gouraud')
        plt.title(f"LOFAR - {self.audio_file}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
        os.makedirs(self.data_path, exist_ok=True)
        plt.savefig(f"{self.data_path}/{output_filename}")
        plt.close()
        print(f"LOFARgram saved as {output_filename}") 

# TODO Pylint ta reclamando da quantidade de variaveis e ifs dessas funcoes, revisar isso aq depois #pylint: disable=fixme
def plot_all_buoy(manager: ArtifactManager, loader: DataLoader, \
                  artifact_type: str, plot_type: str, signal_type: str):
    """ Plots the PSD (Power Spectral Density) for all artifacts of a given type and buoy.

    Args:
        manager (ArtifactManager): Instance of the ArtifactManager class.
        loader (DataLoader): Instance of the DataLoader class.
        artifact_type (str): Type of the artifact to plot (e.g., 'EX-SUP', 'HE3m').
    """

    buoy_data = {}

    for id_artifact in manager.id_from_type(artifact_type):
        for buoy_id, time in manager[id_artifact]:

            if signal_type == 'artifact':
                start_time = time - timedelta(seconds=0.05)
                end_time = time + timedelta(seconds=0.25)
            elif signal_type == 'background':
                start_time = time - timedelta(seconds=10)
                end_time = time - timedelta(seconds=2)

            fs, audio = loader.get_data(buoy_id, start_time, end_time)

            if buoy_id not in buoy_data:
                buoy_data[buoy_id] = []

            buoy_data[buoy_id].append(audio)

    for buoy_id, audio_list in buoy_data.items():
        plt.figure(figsize=(12, 6))
        if signal_type == 'artifact':
            plt.title(f"Artifact {plot_type} for {artifact_type} - Boia {buoy_id}")
        elif signal_type == 'background':
            plt.title(f"Background {plot_type} for {artifact_type} - Boia {buoy_id}")
        plt.xlabel("Frequency (Hz)")
        if plot_type == 'psd':
            plt.ylabel("PSD (W/Hz)")
        elif plot_type == 'fft':
            plt.ylabel("Amplitude (dB)")
        plt.grid()

        for i, audio in enumerate(audio_list):

            if plot_type == 'psd':
                freq, result = lps_bb.psd(signal=audio, fs=fs, window_size=4096, overlap=0.5)
            elif plot_type == 'fft':
                fft_result = np.fft.fft(audio)
                fft_freq = np.fft.fftfreq(len(audio), 1/fs)
                magnitude = np.abs(fft_result)[:len(fft_result)//2]
                magnitude[magnitude == 0] = 1e-10
                result = 20 * np.log10(magnitude)
                freq = fft_freq[:len(fft_result)//2]

            normalization = lps_signal.Normalization.MIN_MAX_ZERO_CENTERED
            result = normalization(result)

            plt.plot(freq, result, label=f"{signal_type} {i+1}")

        plt.legend()
        os.makedirs(f"data/Analysis/{artifact_type}/all/Boia{buoy_id}", exist_ok=True)
        plt.savefig(f"data/Analysis/{artifact_type}/all/Boia{buoy_id}/{plot_type}_all_{signal_type}s.png")
        plt.close()
        print(f"{plot_type} of all artifacts for {artifact_type} for Boia {buoy_id} plotted and saved as {plot_type}_all_{signal_type}s.png")

def plot_artifact_bkg(start_shift: int, bkg_shift: int, end_shift: int):
    '''Plot artifact and background'''

    loader = DataLoader()
    manager = ArtifactManager()

    for artifact_type in ['EX-SUP', 'HE3m', 'GAE']:

        for id_artifact in manager.id_from_type(artifact_type):

            for buoy_id, time in manager[id_artifact]:

                start_time = time - timedelta(seconds=start_shift)
                bkg_end = time - timedelta(seconds=bkg_shift)
                end_time = time + timedelta(seconds=end_shift)

                duration = (end_time - bkg_end).total_seconds()

                fs, audio = loader.get_data(buoy_id, bkg_end, end_time)
                n_samples = int(duration * fs)

                audio_path = f'data/Analysis/{artifact_type}/Boia{buoy_id}/{time}/Artifact'
                bkg_path = f'data/Analysis/{artifact_type}/Boia{buoy_id}/{time}/Background'

                bkg_duration = (bkg_end - start_time).total_seconds()

                bkg_fs, bkg_audio = loader.get_data(buoy_id, start_time, bkg_end)
                bkg_samples = int(bkg_duration * bkg_fs)

                psd_freq, psd_result = lps_bb.psd(signal=audio, fs=fs,\
                                                window_size=4096, overlap=0.5)
                psd_bkg_freq, psd_bkg_result = lps_bb.psd(signal=bkg_audio, fs=bkg_fs,\
                                                            window_size=4096, overlap=0.5)

                data_path = f'data/Analysis/{artifact_type}/{id_artifact}/Boia{buoy_id}/{time}'

                output_filename = 'artifactxbkg_psd.png'

                plt.figure(figsize=(12, 6))
                plt.plot(psd_freq, psd_result, label='Artifact')
                plt.plot(psd_bkg_freq, psd_bkg_result, label='Background')
                plt.title("PSD - Artifact x Background")
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power [W/Hz]')
                plt.grid()
                plt.legend()
                os.makedirs(data_path, exist_ok=True)
                plt.savefig(f"{data_path}/{output_filename}")
                plt.close()
                print(f"PSD saved as {output_filename}")

def artifact_analysis(start_shift: int, bkg_shift: int, end_shift: int):
    """ Analyzes audio data from different artifact types (types of munition) and saves the results.

    This function processes audio associated with different artifacts by extracting
    relevant time segments (time of impact and background noise), performing analysis 
    (waveform, PSD, FFT, and LOFAR), and saving the resulting plots.

    Raises:
        ValueError: If the artifact type does not match any of the expected types.
    """

    loader = DataLoader()
    manager = ArtifactManager()

    for artifact_type in ['EX-SUP', 'HE3m', 'GAE']:

        plot_all_buoy(manager, loader, artifact_type, 'psd', 'artifact')
        plot_all_buoy(manager, loader, artifact_type, 'psd', 'background')
        plot_all_buoy(manager, loader, artifact_type, 'fft', 'artifact')
        plot_all_buoy(manager, loader, artifact_type, 'fft', 'background')

        for id_artifact in manager.id_from_type(artifact_type):

            for buoy_id, time in manager[id_artifact]:

                start_time = time - timedelta(seconds=start_shift)
                bkg_end = time - timedelta(seconds=bkg_shift)
                end_time = time + timedelta(seconds=end_shift)

                artifact_path = f'data/Analysis/{artifact_type}/{id_artifact}/Boia{buoy_id}/Artifact'
                simple_plots(loader,buoy_id,bkg_end,end_time,artifact_path,time,"")

                bkg_path = f'data/Analysis/{artifact_type}/{id_artifact}/Boia{buoy_id}/Background'
                simple_plots(loader,buoy_id,start_time,bkg_end,bkg_path,time,"bkg")

                both_path = f'data/Analysis/{artifact_type}/{id_artifact}/Boia{buoy_id}/Both'
                simple_plots(loader,buoy_id,start_time,end_time,both_path,time,"")

def plot_all_wavs(start_shift: int,end_shift: int):

    loader = DataLoader()
    manager = ArtifactManager()

    for artifact_type in ['EX-SUP', 'HE3m', 'GAE']:
        for id_artifact in manager.id_from_type(artifact_type):
            for buoy_id, time in manager[id_artifact]:

                start_time = time - timedelta(seconds=start_shift)
                end_time = time + timedelta(seconds=end_shift)

                path = f'data/Analysis/{artifact_type}/Boia{buoy_id}'
                simple_plots(loader,buoy_id,start_time,end_time,path,time,id_artifact)

def simple_plots(loader: DataLoader, buoy_id: int, start: datetime, end: datetime, path: str, time: datetime, title: any) -> None:
    duration = (end - start).total_seconds()

    fs, audio = loader.get_data(buoy_id, start, end)
    samples = int(duration * fs)

    analysis = AudioAnalysis(audio, fs, duration, \
                                    samples, path, time)

    analysis.save(f'{title}.wav')
    # analysis.plot(f'{time}_{title}.png')
    # analysis.psd(f'{time}_psd_{title}.png')
    # analysis.fft(f'{time}_fft_{title}.png')
    # analysis.lofar(f'{time}_lofar_{title}.png')

if __name__ == "__main__":

    if os.path.exists("data/Analysis"):
        shutil.rmtree("data/Analysis")
    os.mkdir("data/Analysis")

    START_SHIFT = 2.5
    BKG_SHIFT = 0
    END_SHIFT = 2.5

    plot_all_wavs(START_SHIFT, END_SHIFT)
