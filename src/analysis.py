import wave
import os
import numpy as np
import matplotlib.pyplot as plt

# import lps_sp.signal as lps_signal
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.broadband as lps_bb

class AudioAnalysis:
    def __init__(self, audio_file, buoy_id, audio_id):
        self.audio_file = audio_file
        self.wave_file = None
        self.signal = None
        self.fs = None
        self.time_axis = None
        self.data_path = f'/home/gabriel.lisboa/Workspace/RVT/rvt/data/Analysis/Boia{buoy_id}/{audio_id}'

    def get_audio(self, buoy_id):
        '''Ainda tem que integrar com o loader e o identificador de artefatos'''
        pass

    def load_audio(self):
        """Carrega o arquivo .wav e armazena os dados do áudio"""
        try:
            self.wave_file = wave.open(self.audio_file, 'rb')
            print(f"Arquivo de áudio {self.audio_file} carregado com sucesso.")
            self.extract_signal()
        except wave.Error as e:
            print(f"Erro ao carregar o arquivo de áudio: {e}")

    def extract_signal(self):
        """Extrai os dados do sinal do áudio"""
        if self.wave_file:
            self.fs = self.wave_file.getframerate()
            num_frames = self.wave_file.getnframes()

            audio_frames = self.wave_file.readframes(num_frames)

            self.signal = np.frombuffer(audio_frames, dtype=np.int16)

            duration = num_frames / float(self.fs)
            self.time_axis = np.linspace(0, duration, num=num_frames)

    def plot(self, output_filename):
        """Salva o plot do áudio no tempo como um arquivo .png"""
        if self.signal is not None and self.time_axis is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(self.time_axis, self.signal)
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
        
        psd_freq, psd_result = lps_bb.psd(signal=self.signal, fs=self.fs, window_size=4096, overlap=0.5)

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
        fft_result = np.fft.fft(self.signal)
        fft_freq = np.fft.fftfreq(len(self.signal), 1/self.fs)

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

        power, freq, time = lps_analysis.SpectralAnalysis.lofar(self.signal, self.fs)

        plt.figure()
        plt.pcolormesh(time, freq, 10 * np.log10(power), shading='gouraud')
        plt.title(f"LOFAR - {self.audio_file}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
        os.makedirs(self.data_path, exist_ok=True)
        plt.savefig(f"{self.data_path}/{output_filename}")
        plt.close()

    def close_audio(self):
        """Fecha o arquivo de áudio"""
        if self.wave_file:
            self.wave_file.close()
            print(f"Arquivo de áudio {self.audio_file} fechado.")

if __name__ == "__main__":
    path_to_file = '/home/gabriel.lisboa/Workspace/RVT/Data/RVT/raw_data/20240119/boia1'
    audio_file = '19_10_59_00015.wav'
    audio_id = audio_file.split('.')[0]
    audio = os.path.join(path_to_file, audio_file)
    audio_loader = AudioAnalysis(audio, '1', audio_id)
    audio_loader.load_audio()

    audio_loader.plot(f'{audio_id}.png')
    audio_loader.psd(f'{audio_id}_psd.png')
    audio_loader.fft(f'{audio_id}_fft.png')
    audio_loader.lofar(f'{audio_id}_lofar.png')

    audio_loader.close_audio()