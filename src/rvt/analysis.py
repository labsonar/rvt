import os
import shutil
import typing
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# import lps_sp.signal as lps_signal
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.signal as lps_signal
from rvt.artifact import ArtifactManager
from rvt.loader import DataLoader

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
    def __init__(self, audio: np.ndarray, fs: int, duration: float, n_samples: int,\
        data_path: str, time: datetime):

        self.audio_file = time
        self.audio = audio
        self.fs = fs
        self.duration = duration
        self.n_samples = n_samples
        self.time_axis = np.linspace(0, self.duration, num=self.n_samples)
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

        if self.audio is not None and self.time_axis is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(self.time_axis, self.audio)
            plt.title(f"Audio Waveform - {self.data_path}")
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

    def psd(self, output_filename):
        """ Generates and saves the Power Spectral Density (PSD) plot.

        Args:
            output_filename (str): The filename for saving the plot.
        """
        
        psd_freq, psd_result = lps_bb.psd(signal=self.audio, fs=self.fs, window_size=4096, overlap=0.5)

        plt.figure(figsize=(12, 6))
        plt.plot(psd_freq, psd_result)
        plt.title(f"PSD - {self.data_path}")
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
        plt.title(f"FFT - {self.data_path}")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid()
        os.makedirs(self.data_path, exist_ok=True)
        plt.savefig(f"{self.data_path}/{output_filename}")
        plt.close()
        print(f"FFT saved as {output_filename}")

    def lofar(self, output_filename: str) -> None:
        """ Generates and saves LOFAR analysis plot.

        Args:
            output_filename (str): The filename for saving the plot.
        """

        power, freq, time = lps_analysis.SpectralAnalysis.lofar(self.audio, self.fs)

        plt.figure()
        plt.pcolormesh(time, freq, power, shading='gouraud')
        plt.title(f"LOFAR - {self.data_path}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
        lofar_path = f"{self.data_path}/LOFAR"
        os.makedirs(lofar_path, exist_ok=True)
        plt.savefig(f"{lofar_path}/{output_filename}")
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
        plt.legend()
        path = f"{self.data_path}/{plot_type}"
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{output_filename}")
        plt.close()
        print(f"{plot_type} saved as {output_filename}")
                

def manage_plots(loader: DataLoader, manager: ArtifactManager, start_shift: int, bkg_shift: int, end_shift: int,
                 artifact_types: list, artifact_ids: list, plot_types: list, signal_types: list, compare: bool = False) -> None:

    for artifact_type in artifact_types:
        for artifact_id in artifact_ids:
            if artifact_id in manager.id_from_type(artifact_type):
                for buoy_id in manager[artifact_id]:

                    time = manager.get_time(artifact_id, buoy_id)

                    start_time = time - timedelta(seconds=start_shift)
                    bkg_end = time - timedelta(seconds=bkg_shift)
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
    analysis.plot(f'{time}_{title}.png')
    analysis.psd(f'{time}_psd_{title}.png')
    analysis.fft(f'{time}_fft_{title}.png')
    analysis.lofar(f'{time}_lofar_{title}.png')

def plot(buoy_id: int, artifact_id: int, plot_: str, start: datetime, end: datetime,
        time: datetime, root: str) -> None:
    """ Plot data.

    Args:
        buoy_id (int): Buoy identification
        artifact_id (int): Artifact identification
        plots (typing.List[str]): Plots types
        start (datetime): Start time
        end (datetime): End time
        time (datetime): Offset of detection
        root (str, optional): Root where plot is saved. Defaults to "Analysis".
    """

    manager = ArtifactManager()
    loader = DataLoader()

    artifact_type = manager.type_from_id(artifact_id)
    duration = (end-start).total_seconds()

    path = os.path.join(root,f"{artifact_type}/Buoy {buoy_id}/{artifact_id}")

    fs, audio = loader.get_data(buoy_id, bkg_end, end)
    samples = int(duration * fs)

    analysis = AudioAnalysis(audio, fs, duration, \
                                samples, path, time)

    if "fft" == plot_:
        analysis.fft("fft.png")

    if "lofar" == plot_:
        analysis.lofar("lofar.png")

    if "psd" == plot_:
        analysis.psd("psd.png")

    if "time" == plot_:
        analysis.plot("time.png")

    if "wav" == plot_:
        analysis.save("audio.wav")

if __name__ == "__main__":

    if os.path.exists("data/Analysis"):
        shutil.rmtree("data/Analysis") # Usando so para testes
    os.makedirs("data/Analysis")
    
    START_SHIFT = 10
    BKG_SHIFT = 2
    END_SHIFT = 2

    manager_ = ArtifactManager()
    loader_ = DataLoader()

    artifact_ids = []
    for artifact_id in args.artifact_ids:
        if "-" in artifact_id:
            start, end = map(int, artifact_id.split("-"))
            artifact_ids.extend(range(start, end + 1))
        else:
            artifact_ids.append(int(artifact_id))
            
    manage_plots(loader_, manager_, START_SHIFT, BKG_SHIFT, END_SHIFT, args.artifact_types,
                 artifact_ids, args.plot_types, args.signal_type, args.compare)

if __name__ == "__main__":
    
    main()