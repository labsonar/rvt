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
    def __init__(self, audio, fs, duration, n_samples, data_path, time):
        self.audio_file = time
        # self.wave_file = None
        self.audio = audio
        self.fs = fs
        self.duration = duration
        self.n_samples = n_samples
        self.time_axis = None
        self.data_path = data_path
        # self.data_path = f'../data/Analysis/{artifact_type}/Boia{buoy_id}/{audio_id}'

    # def load_audio(self):
    #     """Carrega o arquivo .wav e armazena os dados do áudio"""
    #     try:
    #         self.wave_file = wave.open(self.audio_file, 'rb')
    #         print(f"Arquivo de áudio {self.audio_file} carregado com sucesso.")
    #         self.extract_signal()
    #     except wave.Error as e:
    #         print(f"Erro ao carregar o arquivo de áudio: {e}")

    # def extract_signal(self):
    #     """Extrai os dados do sinal do áudio"""
    #     if self.wave_file:
    #         self.fs = self.wave_file.getframerate()
    #         num_frames = self.wave_file.getnframes()

    #         audio_frames = self.wave_file.readframes(num_frames)

    #         self.audio = np.frombuffer(audio_frames, dtype=np.int16)

    #         duration = num_frames / float(self.fs)
    #         self.time_axis = np.linspace(0, duration, num=num_frames)

    def plot(self, output_filename):
        """Salva o plot do áudio no tempo como um arquivo .png"""

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
            print(f"Gráfico salvo como {output_filename}")
        else:
            print("Nenhum áudio foi carregado ou extraído.")
        
    def psd(self, output_filename):
        
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

    def fft(self, output_filename):
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

    def lofar(self, output_filename):

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

    # def close_audio(self):
    #     """Fecha o arquivo de áudio"""
    #     if self.wave_file:
    #         self.wave_file.close()
    #         print(f"Arquivo de áudio {self.audio_file} fechado.")

def artifact_analysis():

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

            
artifact_analysis()
# if __name__ == "__main__":

#     path_to_file = '/home/gabriel.lisboa/Workspace/RVT/Data/RVT/raw_data/20240119/boia1'
#     audio_file = '19_10_59_00015.wav'
#     audio_id = audio_file.split('.')[0]
#     audio = os.path.join(path_to_file, audio_file)
#     audio_loader = AudioAnalysis(audio, '1', audio_id)
#     audio_loader.load_audio()

#     audio_loader.plot(f'{audio_id}.png')
#     audio_loader.psd(f'{audio_id}_psd.png')
#     audio_loader.fft(f'{audio_id}_fft.png')
#     audio_loader.lofar(f'{audio_id}_lofar.png')

#     audio_loader.close_audio()