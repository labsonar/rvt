import wave
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# import lps_sp.signal as lps_signal
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.broadband as lps_bb
from artifact import ArtifactManager
from loader import DataLoader

class AudioAnalysis:
    """ Class for audio analysis operations.

    This class provides methods for plotting waveforms, computing Power Spectral Density (PSD), 
    performing Fast Fourier Transform (FFT), and generating LOFAR analysis plots from an audio signal.
    
    Attributes:
        audio (ndarray): The audio data loaded for analysis.
        fs (int): The sampling frequency of the audio data.
        duration (float): Duration of the audio signal.
        n_samples (int): Number of samples in the audio signal.
        data_path (str): Path to save analysis results.
        time (datetime): Time associated with the audio file for reference.
    """

    def __init__(self, audio, fs, duration, n_samples, data_path, time):
        self.audio_file = time
        self.audio = audio
        self.fs = fs
        self.duration = duration
        self.n_samples = n_samples
        self.time_axis = None
        self.data_path = data_path

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

        plt.figure()
        plt.plot(fft_freq[:len(fft_result)//2], 
                 20*np.log10(np.abs(fft_result)[:len(fft_result)//2]))
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
        plt.pcolormesh(time, freq, 10 * np.log10(power), shading='gouraud')
        plt.title(f"LOFAR - {self.audio_file}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
        os.makedirs(self.data_path, exist_ok=True)
        plt.savefig(f"{self.data_path}/{output_filename}")
        plt.close()
        print(f"LOFARgram saved as {output_filename}")


def artifact_analysis():
    """ Analyzes audio data from different artifact types (types of munition) and saves the results.

    This function processes audio associated with different artifacts by extracting
    relevant time segments (time of impact and background noise), performing analysis 
    (waveform, PSD, FFT, and LOFAR), and saving the resulting plots.

    Raises:
        ValueError: If the artifact type does not match any of the expected types.
    """

    artifact_type = ''
    manager = ArtifactManager(base_path="../data/artifacts.csv")

    for id_artifact in manager:
        
        if id_artifact in manager.id_from_type('EX-SUP'):
            artifact_type = 'EX-SUP'
        elif id_artifact in manager.id_from_type('HE3m'):
            artifact_type = 'HE3m'
        elif id_artifact in manager.id_from_type('GAE'):
            artifact_type = 'GAE'
        else:
            raise ValueError(f"Artifact #{id_artifact} is not of any expected type")

        for buoy_id, time in manager[id_artifact]:

            start_time = time - timedelta(seconds=10)
            bkg_end = time - timedelta(seconds=2)
            end_time = time + timedelta(seconds=2)

            duration = (end_time - bkg_end).total_seconds()

            loader = DataLoader("../../Data/RVT/raw_data")
            
            fs, audio = loader.get_data(buoy_id, bkg_end, end_time)
            n_samples = int(duration * fs)

            audio_path = f'../data/Analysis/{artifact_type}/Boia{buoy_id}/{time}/Artifact'
            bkg_path = f'../data/Analysis/{artifact_type}/Boia{buoy_id}/{time}/Background'

            audio_analysis = AudioAnalysis(audio, fs, duration, n_samples, audio_path, time)

            audio_analysis.plot(f'{time}.png')
            audio_analysis.psd(f'{time}_psd.png')
            audio_analysis.fft(f'{time}_fft.png')
            audio_analysis.lofar(f'{time}_lofar.png')

            bkg_duration = (bkg_end - start_time).total_seconds()

            bkg_fs, bkg_audio = loader.get_data(buoy_id, start_time, bkg_end)
            bkg_samples = int(bkg_duration * bkg_fs)

            bkg_analysis = AudioAnalysis(bkg_audio, bkg_fs, bkg_duration, bkg_samples, bkg_path, time)

            bkg_analysis.plot(f'{time}_bkg.png')
            bkg_analysis.psd(f'{time}_psd_bkg.png')
            bkg_analysis.fft(f'{time}_fft_bkg.png')
            bkg_analysis.lofar(f'{time}_lofar_bkg.png')

            
if __name__ == "__main__":

    artifact_analysis()