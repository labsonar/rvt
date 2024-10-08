import wave
import numpy as np
import matplotlib.pyplot as plt

import lps_sp.signal as lps_signal
import lps_sp.pdf as lps_pdf
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.broadband as lps_bb

class AudioLoader:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.wave_file = None
        self.signal = None
        self.frame_rate = None
        self.fs = 48000
        self.time_axis = None

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
            # Número de canais e taxa de amostragem (frames per second)
            self.frame_rate = self.wave_file.getframerate()
            num_frames = self.wave_file.getnframes()

            # Lê o áudio como um array de bytes
            audio_frames = self.wave_file.readframes(num_frames)

            # Converte os dados do áudio para um array numpy
            self.signal = np.frombuffer(audio_frames, dtype=np.int16)

            # Cria o eixo do tempo
            duration = num_frames / float(self.frame_rate)
            self.time_axis = np.linspace(0, duration, num=num_frames)

    def save_audio_plot(self, output_filename):
        """Salva o plot do áudio no tempo como um arquivo .png"""
        if self.signal is not None and self.time_axis is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(self.time_axis, self.signal)
            plt.title(f"Audio Waveform - {self.audio_file}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.savefig(output_filename)
            plt.close()  # Fecha o plot após salvar para liberar memória
            print(f"Gráfico salvo como {output_filename}")
        else:
            print("Nenhum áudio foi carregado ou extraído.")

    def windowing(self, start_time1, duration1, start_time2, duration2):
        """Extrai duas janelas do áudio e retorna os dados e o tempo correspondentes"""
        if self.signal is not None and self.time_axis is not None:
            # Calcular os índices das janelas
            start_index1 = int(start_time1 * self.frame_rate)
            end_index1 = int((start_time1 + duration1) * self.frame_rate)
            window1_signal = self.signal[start_index1:end_index1]
            window1_time = self.time_axis[start_index1:end_index1]

            start_index2 = int(start_time2 * self.frame_rate)
            end_index2 = int((start_time2 + duration2) * self.frame_rate)
            window2_signal = self.signal[start_index2:end_index2]
            window2_time = self.time_axis[start_index2:end_index2]

            return (window1_time, window1_signal), (window2_time, window2_signal)

    # def pdf(self, start_time1, duration1, start_time2, duration2):
    #     (window1_time, window1_signal), (window2_time, window2_signal) = self.windowing(start_time1, duration1, start_time2, duration2)

    #     lps_pdf.plot_pdf('pdf.png', window1_signal, window2_signal, n_bins=50,
    #          title1 = "Window 1", title2 = "Window 2")
        
    def psd(self, output_filename):
        
        psd_freq, psd_result = lps_bb.psd(signal=self.signal, fs=self.fs, window_size=4096, overlap=0.5)

        plt.figure(figsize=(12, 6))
        plt.plot(psd_freq, psd_result, label='Test Spectrum')
        # plt.plot(frequencies, desired_spectrum, linestyle='--', label='Desired Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.savefig(output_filename)
        plt.close()

    def fft(self, output_filename):
        # Calcula a FFT do sinal
        fft_data = np.fft.fft(self.signal)
        frequency_data = np.fft.fftfreq(len(self.signal), 1/self.fs)

        plt.figure()
        # plt.plot(frequency_data[:len(fft_data)//2], 
        #          np.abs(fft_data)[:len(fft_data)//2])
        plt.plot(frequency_data, np.abs(fft_data))
        plt.title('FFT of the Signal')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.savefig(output_filename)
        plt.close()

    def lofar(self, output_filename):

        power, freq, time = lps_analysis.SpectralAnalysis.lofar(self.signal, self.fs)

        plt.figure()
        plt.pcolormesh(time, freq, 10 * np.log10(power), shading='gouraud')
        plt.title('Lofar')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
        plt.savefig(output_filename)
        plt.close()

    def close_audio(self):
        """Fecha o arquivo de áudio"""
        if self.wave_file:
            self.wave_file.close()
            print(f"Arquivo de áudio {self.audio_file} fechado.")

# Exemplo de uso
audio_file = '/home/gabriel.lisboa/Workspace/RVT/Data/RVT/raw_data/20240119/boia1/19_10_59_00015.wav'
audio_loader = AudioLoader(audio_file)
audio_loader.load_audio()

# Plotar o áudio no tempo
print('plot')
audio_loader.save_audio_plot('teste.png')

audio_loader.psd('psd_test.png')
audio_loader.fft('fft_test.png')
audio_loader.lofar('lofar.png')

# Fechar o arquivo de áudio
audio_loader.close_audio()