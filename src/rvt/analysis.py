import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

#import lps_sp.acoustical.analysis as lps_analysis
#import lps_sp.acoustical.broadband as lps_bb


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

    def __init__(self, data_path, time):
        self.audio_file = time
        self.data_path = data_path

    def save(self, fs, audio, output_filename: str) -> None:
        """Save the audio in .wav format

        Args:
            output_filename (str): Name of saved file.
        """

        os.makedirs(self.data_path, exist_ok=True)
        path = os.path.join(self.data_path, output_filename)
        if audio is not None :
            scipy.io.wavfile.write(path, fs, audio)
            print(f"Audio saved as {output_filename}")
        else :
            print("Nenhum áudio foi carregado ou extraído.")

    def waveform(self, audio, fs, output_filename):
        """ Generates and saves the waveform plot of the audio.

        Args:
            output_filename (str): The filename for saving the plot.
        """

        n_samples = len(audio)
        duration= n_samples / fs
        time_axis = np.linspace(0, duration, num=n_samples)

        if audio is not None and time_axis is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(self.time_axis, self.audio)
            plt.title(f"Audio Waveform - {self.audio_file}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.grid()
            waveform_path = f"{self.data_path}/Waveform"
            os.makedirs(waveform_path, exist_ok=True)
            plt.savefig(f"{waveform_path}/{output_filename}")
            plt.close()
            print(f"Audio Waveform saved as {output_filename}")
        else:
            print("Nenhum áudio foi carregado ou extraído.")

    def psd(self, audio, fs):
        """
        Computes the Power Spectral Density (PSD) of the audio signal.

        Returns:
            tuple: Frequencies and PSD values.
        """
    
        psd_freq, psd_result = lps_bb.psd(signal=audio, fs=fs, window_size=4096, overlap=0.5)

        return psd_freq, psd_result

    def fft(self, audio, fs):
        """
        Computes the Fast Fourier Transform (FFT) of the audio signal.
        
        Returns:
            tuple: FFT frequencies and magnitude in dB.
        """

        fft_result = np.fft.fft(audio)
        fft_freq = np.fft.fftfreq(len(audio), 1/fs)

        magnitude = np.abs(fft_result)[:len(fft_result)//2]
        magnitude[magnitude == 0] = 1e-10

        return fft_freq[:len(fft_result)//2], 20*np.log10(magnitude)

    def lofar(self, audio, fs, output_filename):
        """
        Generates and saves LOFAR analysis plot.

        Args:
            output_filename (str): The filename for saving the plot.
        """

        power, freq, time = lps_analysis.SpectralAnalysis.lofar(audio, fs)

        plt.figure()
        plt.pcolormesh(time, freq, power, shading='gouraud')
        plt.title(f"LOFAR - {self.audio_file}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
        lofar_path = f"{self.data_path}/LOFAR"
        os.makedirs(lofar_path, exist_ok=True)
        plt.savefig(f"{lofar_path}/{output_filename}")
        plt.close()
        print(f"LOFARgram saved as {output_filename}")

    def plot(self, x1, y1, plot_type, xlabel, ylabel, output_filename,
             x2=None, y2=None, compare=False):
        """
        Generate and save a customizable plot of the audio signal.
        
        Args:
            x1 (array-like): Data for the x-axis.
            y1 (array-like): Data for the y-axis.
            plot_type (str): Type of plot (e.g., 'PSD', 'FFT').
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            output_filename (str): Filename to save the plot.
            x2 (array-like, optional): Secondary data for the x-axis (for comparison).
            y2 (array-like, optional): Secondary data for the y-axis (for comparison).
            compare (bool, optional): Flag to enable comparison plotting.
        """
        plt.figure(figsize=(12,6))
        plt.plot(x1, y1)

        if compare:
            plt.plot(x2, y2)
            title = f"{plot_type} - Artifact x BKG - {self.audio_file}"
        else:
            title = f"{plot_type} - {self.audio_file}"

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        path = os.path.join(self.data_path, plot_type)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, output_filename))
        plt.close()
        print(f"{plot_type} saved as {output_filename}")
        