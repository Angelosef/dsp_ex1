import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT

# part2

# 2.1
# load speech_utterance.wav and plot it

directory = '..\\data\\speech_utterance.wav'
sampling_frequency, speech_signal = wavfile.read(directory)


time = np.arange(len(speech_signal)) / sampling_frequency
plt.plot(time, speech_signal)

plt.xlabel('time (s)')
plt.ylabel('Amplitude')

plt.title('Speech Utterance Signal')

plt.show()


# 2.2
# calculating energy and sign switching rate of signal

number_of_window_lengths = 3
window_length_time = np.linspace(20, 50, number_of_window_lengths) / 1000

def create_window(time_length, sampling_frequency):
    window_length = round(time_length * sampling_frequency)
    return np.hamming(window_length)

def get_short_time_energy(signal, window):
    squared_signal = np.zeros(len(signal))
    for i in range(len(squared_signal)):
        squared_signal[i] = signal[i] ** 2
    
    return np.convolve(squared_signal, window)

energy_signal = np.empty((number_of_window_lengths,), dtype=object)
for i in range(number_of_window_lengths):
    window = create_window(window_length_time[i], sampling_frequency)
    sig = get_short_time_energy(speech_signal, window)
    energy_signal[i] = sig


plots = number_of_window_lengths + 1
fig, axs = plt.subplots(plots, 1, figsize=(8, 6), sharex=True)

time = np.arange(len(speech_signal)) / sampling_frequency
axs[0].plot(time, speech_signal)
axs[0].set_title('Speech Utterance Signal')

for i in range(number_of_window_lengths):
    time = window_length_time[i]
    string = f'short time energy with window length {time}s'
    time = np.arange(len(energy_signal[i])) / sampling_frequency
    axs[i+1].set_title(string)
    axs[i+1].plot(time, energy_signal[i])

axs[-1].set_xlabel('time (s)')
plt.tight_layout()
plt.show()


def get_sign_switching_rate(signal, window):
    signal_sign = np.sign(signal)
    sign_change = np.zeros(len(signal))
    sign_change[0] = np.abs(signal_sign[0])
    for i in range(1, len(sign_change)):
        value = signal_sign[i] - signal_sign[i-1]
        sign_change[i] = np.abs(value)
    
    return np.convolve(sign_change, window)

switch_rate_signal = np.empty((number_of_window_lengths,), dtype=object)
for i in range(number_of_window_lengths):
    window = create_window(window_length_time[i], sampling_frequency)
    sig = get_sign_switching_rate(speech_signal, window)
    switch_rate_signal[i] = sig


plots = number_of_window_lengths + 1
fig, axs = plt.subplots(plots, 1, figsize=(8, 6), sharex=True)

time = np.arange(len(speech_signal)) / sampling_frequency
axs[0].plot(time, speech_signal)
axs[0].set_title('Speech Utterance Signal')
for i in range(number_of_window_lengths):
    time = window_length_time[i]
    string = f'switch rate with window length {time}s'
    time = np.arange(len(switch_rate_signal[i])) / sampling_frequency
    axs[i+1].set_title(string)
    axs[i+1].plot(time, switch_rate_signal[i])

axs[-1].set_xlabel('time (s)')
plt.tight_layout()
plt.show()



window = create_window(0.025, sampling_frequency)
energy = get_short_time_energy(speech_signal, window)
switch_sign_rate = get_sign_switching_rate(speech_signal, window)

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
time = np.arange(len(speech_signal)) / sampling_frequency
axs[0].plot(time, speech_signal)
axs[0].set_title('Speech Utterance Signal')
time = np.arange(len(energy)) / sampling_frequency
axs[1].plot(time, energy)
axs[1].set_title('short time energy')
time = np.arange(len(switch_sign_rate)) / sampling_frequency
axs[2].plot(time, switch_sign_rate)
axs[2].set_title('switching sign rate')

axs[-1].set_xlabel('time (s)')
plt.tight_layout()

plt.show()


# 2.3
# short time fourier transform
window = create_window(0.035, sampling_frequency)
fft_points = 2048
hop_length = round(len(window) / 2)
sft = ShortTimeFFT(window, hop_length, sampling_frequency, mfft=fft_points)
stft_of_signal = sft.stft(speech_signal)


plt.figure(figsize=(10, 6))
param1 = 20 * np.log10(np.abs(stft_of_signal))
param2 = sft.extent(len(speech_signal))
plt.imshow(param1, aspect='auto', extent = param2, origin='lower', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of Speech Signal')
plt.show()


def get_spectral_centroid(stft_of_signal, sampling_frequency):
    shape = np.shape(stft_of_signal)
    spectral_centroid = np.zeros(shape[1])
    N = 2 * shape[0]
    for m in range(shape[1]):
        coefficient = 0
        for k in range(shape[0]):
            value = np.abs(stft_of_signal[k][m]) ** 2
            spectral_centroid[m] += k * value
            coefficient += value
        
        coefficient = (1 / coefficient) * sampling_frequency / N
        spectral_centroid[m] = coefficient * spectral_centroid[m]
    
    return spectral_centroid

def get_spectral_flux(stft_of_signal):
    shape = np.shape(stft_of_signal)
    flux_length = shape[1] - 1
    spectral_flux = np.zeros(flux_length)
    const1 = np.zeros(flux_length)
    const2 = np.zeros(flux_length)
    for m in range(flux_length):
        for k in range(shape[0]):
            const1[m] += np.abs(stft_of_signal[k][m+1]) ** 2
            const2[m] += np.abs(stft_of_signal[k][m]) ** 2

    for m in range(flux_length):
        vector = np.zeros(shape[0])
        for k in range(shape[0]):
            value1 = np.abs(stft_of_signal[k][m+1]) ** 2
            value2 = np.abs(stft_of_signal[k][m]) ** 2
            vector[k] = (value1 / const1[m]) - (value2 / const2[m])
        spectral_flux[m] = np.linalg.norm(vector)

    return spectral_flux

spectral_centroid = get_spectral_centroid(stft_of_signal, sampling_frequency)
spectral_flux = get_spectral_flux(stft_of_signal)


total_time = len(speech_signal) / sampling_frequency
time = np.linspace(0, total_time, len(spectral_centroid))
plt.plot(time, spectral_centroid)
plt.xlabel('time (s)')
plt.ylabel('frequency (Hz)')
plt.title('Spectral Centroid')
plt.show()

total_time = len(speech_signal) / sampling_frequency
time = np.linspace(0, total_time, len(spectral_flux))
plt.plot(time, spectral_flux)
plt.xlabel('time (s)')
plt.ylabel('flux amplitude')
plt.title('Spectral Flux')

plt.show()


# 2.4
# again with music.wav

directory = '..\\data\\music.wav'
sampling_frequency, music_signal = wavfile.read(directory)


time = np.arange(len(music_signal)) / sampling_frequency
plt.plot(time, music_signal)

plt.xlabel('time (s)')
plt.ylabel('Amplitude')

plt.title('Music Signal')

plt.show()


number_of_window_lengths = 3
window_length_time = np.linspace(20, 50, number_of_window_lengths) / 1000

energy_signal = np.empty((number_of_window_lengths,), dtype=object)
for i in range(number_of_window_lengths):
    window = create_window(window_length_time[i], sampling_frequency)
    sig = get_short_time_energy(music_signal, window)
    energy_signal[i] = sig


plots = number_of_window_lengths + 1
fig, axs = plt.subplots(plots, 1, figsize=(8, 6), sharex=True)

time = np.arange(len(music_signal)) / sampling_frequency
axs[0].plot(time, music_signal)
axs[0].set_title('Music Signal')

for i in range(number_of_window_lengths):
    time = window_length_time[i]
    string = f'short time energy with window length {time}s'
    time = np.arange(len(energy_signal[i])) / sampling_frequency
    axs[i+1].set_title(string)
    axs[i+1].plot(time, energy_signal[i])

axs[-1].set_xlabel('time (s)')
plt.tight_layout()
plt.show()


switch_rate_signal = np.empty((number_of_window_lengths,), dtype=object)
for i in range(number_of_window_lengths):
    window = create_window(window_length_time[i], sampling_frequency)
    sig = get_sign_switching_rate(music_signal, window)
    switch_rate_signal[i] = sig


plots = number_of_window_lengths + 1
fig, axs = plt.subplots(plots, 1, figsize=(8, 6), sharex=True)

time = np.arange(len(music_signal)) / sampling_frequency
axs[0].plot(time, music_signal)
axs[0].set_title('Music Signal')
for i in range(number_of_window_lengths):
    time = window_length_time[i]
    string = f'switch rate with window length {time}s'
    time = np.arange(len(switch_rate_signal[i])) / sampling_frequency
    axs[i+1].set_title(string)
    axs[i+1].plot(time, switch_rate_signal[i])

axs[-1].set_xlabel('time (s)')
plt.tight_layout()
plt.show()



window = create_window(0.025, sampling_frequency)
energy = get_short_time_energy(music_signal, window)
switch_sign_rate = get_sign_switching_rate(music_signal, window)

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
time = np.arange(len(music_signal)) / sampling_frequency
axs[0].plot(time, music_signal)
axs[0].set_title('Music Signal')
time = np.arange(len(energy)) / sampling_frequency
axs[1].plot(time, energy)
axs[1].set_title('short time energy')
time = np.arange(len(switch_sign_rate)) / sampling_frequency
axs[2].plot(time, switch_sign_rate)
axs[2].set_title('switching sign rate')

axs[-1].set_xlabel('time (s)')
plt.tight_layout()

plt.show()


window = create_window(0.025, sampling_frequency)
fft_points = 2048
hop_length = round(len(window) / 2)
sft = ShortTimeFFT(window, hop_length, sampling_frequency, mfft=fft_points)
stft_of_signal = sft.stft(music_signal)


plt.figure(figsize=(10, 6))
param1 = 20 * np.log10(np.abs(stft_of_signal))
param2 = sft.extent(len(music_signal))
plt.imshow(param1, aspect='auto', extent = param2, origin='lower', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of Music Signal')
plt.show()


spectral_centroid = get_spectral_centroid(stft_of_signal, sampling_frequency)
spectral_flux = get_spectral_flux(stft_of_signal)


total_time = len(speech_signal) / sampling_frequency
time = np.linspace(0, total_time, len(spectral_centroid))
plt.plot(time, spectral_centroid)
plt.xlabel('time (s)')
plt.ylabel('frequency (Hz)')
plt.title('Spectral Centroid')
plt.show()

total_time = len(speech_signal) / sampling_frequency
time = np.linspace(0, total_time, len(spectral_flux))
plt.plot(time, spectral_flux)
plt.xlabel('time (s)')
plt.ylabel('flux amplitude')
plt.title('Spectral Flux')

plt.show()

