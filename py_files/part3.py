import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
from scipy.signal import ShortTimeFFT
from scipy.signal import butter, lfilter
import librosa

# part3

# 3.1
# loading whale signals and transforming them from
# voltage signals to pressure signals

path_to_sperm = '..\\data\\85005006.wav'
path_to_humpback = '..\\data\\9220100Q.wav'

sperm_voltage_signal, sampling_rate = librosa.load(path_to_sperm)
humpback_voltage_signal, sampling_rate = librosa.load(path_to_humpback)

gain_factor = 0.16
sensitivity = -155

def dB_to_value(number):
    return np.power(10, number / 20)

def voltage_to_pressure(voltage_signal, gain_factor, sensitivity):
    pressure_signal = voltage_signal * 3.5 * gain_factor
    pressure_signal = pressure_signal / dB_to_value(sensitivity)

    return pressure_signal

sperm_pressure_signal = voltage_to_pressure(sperm_voltage_signal, gain_factor, sensitivity)
humpback_pressure_signal = voltage_to_pressure(humpback_voltage_signal, gain_factor, sensitivity)

total_time = len(sperm_pressure_signal) / sampling_rate
sperm_time = np.linspace(0, total_time, len(sperm_pressure_signal))

total_time = len(humpback_pressure_signal) / sampling_rate
humpback_time = np.linspace(0, total_time, len(humpback_pressure_signal))


plt.plot(sperm_time, sperm_pressure_signal)
plt.xlabel('time (s)')
plt.ylabel('pressure')
plt.title('Sperm Whale Pressure Signal')
plt.show()

plt.plot(humpback_time, humpback_pressure_signal)
plt.xlabel('time (s)')
plt.ylabel('pressure')
plt.title('Humpback Whale Pressure Signal')
plt.show()


# 3.2
# calculating rms pressure and sound pressure level

sqrt_N = np.sqrt(len(sperm_pressure_signal))
sperm_rms_pressure = np.linalg.norm(sperm_pressure_signal / sqrt_N)

sqrt_N = np.sqrt(len(humpback_pressure_signal))
humpback_rms_pressure = np.linalg.norm(humpback_pressure_signal / sqrt_N)

p_ref = 10 ** -6
sperm_SPL_rms = 20 * np.log10(sperm_rms_pressure / p_ref)
humpback_SPL_rms = 20 * np.log10(humpback_rms_pressure / p_ref)


print('sperm rms pressure = ', sperm_rms_pressure)
print('humpback rms pressure = ', humpback_rms_pressure)

print('sperm SPL rms = ', sperm_SPL_rms)
print('humpback SPL rms = ', humpback_SPL_rms)


# 3.3
# verify parseval theorem

energy_time = np.linalg.norm(sperm_pressure_signal)
energy_time = energy_time ** 2

fft_of_signal = np.fft.fft(sperm_pressure_signal)
energy_frequency = np.abs(fft_of_signal)
energy_frequency = np.linalg.norm(energy_frequency)
energy_frequency = energy_frequency ** 2
energy_frequency = energy_frequency / len(sperm_pressure_signal)


print('energy in the time domain = ', energy_time)
print('energy in the frequency domain = ', energy_frequency)


# 3.4
# filtering out low frequencies from pressure signals

order = 3
cutoff_frequency = 200

num, denom = butter(order, cutoff_frequency, btype='high', analog=False, fs=sampling_rate)

filtered_sperm = lfilter(num, denom, sperm_pressure_signal)
filtered_humpback = lfilter(num, denom, humpback_pressure_signal)


fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(sperm_time, sperm_pressure_signal)
axs[0].set_title('unfiltered sperm whale signal')
axs[0].set_ylabel('pressure')
axs[1].plot(sperm_time, filtered_sperm)
axs[1].set_title('filtered sperm whale signal')
axs[1].set_ylabel('pressure')
axs[-1].set_xlabel('time (s)')

plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(humpback_time, humpback_pressure_signal)
axs[0].set_title('unfiltered humpback whale signal')
axs[0].set_ylabel('pressure')
axs[1].plot(humpback_time, filtered_humpback)
axs[1].set_title('filtered humpback whale signal')
axs[1].set_ylabel('pressure')
axs[-1].set_xlabel('time (s)')

plt.show()




# 3.5
# checking if whales get affected by the sound of pile driving

path_to_pile_driving = '..\\data\\Pile driving.wav'
pile_driving_voltage, sampling_rate = librosa.load(path_to_pile_driving)

gain_factor = 0.16
sensitivity = -175

pile_driving_pressure = voltage_to_pressure(pile_driving_voltage, gain_factor, sensitivity)

sqrt_N = np.sqrt(len(pile_driving_pressure))
pile_driving_rms_pressure = np.linalg.norm(pile_driving_pressure / sqrt_N)

pile_driving_SPL_rms = 20 * np.log10(pile_driving_rms_pressure / p_ref)


print('pile pile driving SPL rms = ', pile_driving_SPL_rms)

if pile_driving_SPL_rms > 100:
    print('pile driving SPL rms value exceeds 100 dB relative to 1uPa')



# 3.6
# calculating short-tme energy of whale signals

def create_window(time_length, sampling_frequency):
    window_length = round(time_length * sampling_frequency)
    return np.hamming(window_length)

def get_short_time_energy(signal, window):
    squared_signal = np.zeros(len(signal))
    for i in range(len(squared_signal)):
        squared_signal[i] = signal[i] ** 2
    
    return np.convolve(squared_signal, window)

time_length = 0.05
window = create_window(time_length, sampling_rate)

sperm_energy = get_short_time_energy(sperm_pressure_signal, window)
humpback_energy = get_short_time_energy(humpback_pressure_signal, window)


time = np.linspace(0, sperm_time[-1], len(sperm_energy))
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(sperm_time, sperm_pressure_signal)
axs[0].set_title('sperm whale signal')
axs[0].set_ylabel('pressure')
axs[1].plot(time, sperm_energy)
axs[1].set_title('short time energy sperm whale signal')
axs[1].set_ylabel('energy')
axs[-1].set_xlabel('time (s)')

plt.show()

time = np.linspace(0, humpback_time[-1], len(humpback_energy))
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(humpback_time, humpback_pressure_signal)
axs[0].set_title('humpback whale signal')
axs[0].set_ylabel('pressure')
axs[1].plot(time, humpback_energy)
axs[1].set_title('short time energy humpback whale signal')
axs[1].set_ylabel('energy')
axs[-1].set_xlabel('time (s)')

plt.show()


# 3.7
# appying teager kaiser energy operator to pressure signals

def teager_kaiser(signal):
    teager_energy = np.zeros(len(signal))

    teager_energy[0] = signal[0] * signal[0]
    teager_energy[0] += -1 * signal[0] * signal[1]

    teager_energy[-1] = signal[-1] * signal[-1]
    teager_energy[-1] += -1 * signal[-2] * signal[-1]

    for n in range(1, len(signal) - 1):
        teager_energy[n] = signal[n] * signal[n]
        teager_energy[n] += -1 * signal[n-1] * signal[n+1]
    
    return teager_energy


sperm_teager_energy = teager_kaiser(sperm_pressure_signal)
humpback_teager_energy = teager_kaiser(humpback_pressure_signal)


time = np.linspace(0, sperm_time[-1], len(sperm_teager_energy))
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(sperm_time, sperm_pressure_signal)
axs[0].set_title('sperm whale signal')
axs[0].set_ylabel('pressure')
axs[1].plot(time, sperm_teager_energy)
axs[1].set_title('teager energy sperm whale signal')
axs[1].set_ylabel('energy')
axs[-1].set_xlabel('time (s)')

plt.show()

time = np.linspace(0, humpback_time[-1], len(humpback_teager_energy))
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(humpback_time, humpback_pressure_signal)
axs[0].set_title('humpback whale signal')
axs[0].set_ylabel('pressure')
axs[1].plot(time, humpback_teager_energy)
axs[1].set_title('teager energy humpback whale signal')
axs[1].set_ylabel('energy')
axs[-1].set_xlabel('time (s)')

plt.show()


# 3.8
# calculating welch signal

sperm_freqs, sperm_power_spectrum = signal.welch(sperm_pressure_signal, sampling_rate)
humpback_freqs, humpback_power_spectrum = signal.welch(humpback_pressure_signal, sampling_rate)


plt.plot(sperm_freqs, sperm_power_spectrum)
plt.xlabel('frequency (Hz)')
plt.ylabel('dB relative to 1uPa^2/Hz')
plt.title('sperm whale welch signal')
plt.xscale('log')

plt.show()

plt.plot(humpback_freqs, humpback_power_spectrum)
plt.xlabel('frequency (Hz)')
plt.ylabel('dB relative to 1uPa^2/Hz')
plt.title('humpback whale welch signal')
plt.xscale('log')

plt.show()


# 3.9
# plotting spectogram of whale pressure signals

sperm_stft = librosa.stft(sperm_pressure_signal, hop_length=64)
sperm_stft_result_dB = librosa.amplitude_to_db(np.abs(sperm_stft), ref=np.max)

humpback_stft = librosa.stft(humpback_pressure_signal, hop_length=64)
humpback_stft_result_dB = librosa.amplitude_to_db(np.abs(humpback_stft), ref=np.max)


fig, ax = plt.subplots()
librosa.display.specshow(sperm_stft_result_dB, sr=sampling_rate, x_axis='time', y_axis='log', hop_length=64)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of sperm whale signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

fig, ax = plt.subplots()
librosa.display.specshow(humpback_stft_result_dB, sr=sampling_rate, x_axis='time', y_axis='log', hop_length=64)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of humpback whale signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()


# 3.10
# using PCEN to supress background noise

sperm_mel = librosa.feature.melspectrogram(S=np.abs(sperm_stft)**2, sr=sampling_rate)
sperm_mel_dB = librosa.power_to_db(sperm_mel, ref=np.max)

pcen_sperm = librosa.pcen(sperm_mel * (2**31))


fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
img = librosa.display.specshow(sperm_mel_dB, x_axis='time', y_axis='mel', ax=ax[0], sr=sampling_rate, hop_length=64)
ax[0].set(title='log amplitude (dB) (sperm whale)', xlabel=None)
ax[0].label_outer()
imgpcen = librosa.display.specshow(pcen_sperm, x_axis='time', y_axis='mel', ax=ax[1], sr=sampling_rate, hop_length=64)
ax[1].set(title='Per-channel energy normalization (sperm whale)')

fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
fig.colorbar(imgpcen, ax=ax[1])

plt.show()


humpback_mel = librosa.feature.melspectrogram(S=np.abs(humpback_stft)**2, sr=sampling_rate)
humpback_mel_dB = librosa.power_to_db(humpback_mel, ref=np.max)

pcen_humpback = librosa.pcen(humpback_mel * (2**31))

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
img = librosa.display.specshow(humpback_mel_dB, x_axis='time', y_axis='mel', ax=ax[0], sr=sampling_rate, hop_length=64)
ax[0].set(title='log amplitude (dB) (humpback whale)', xlabel=None)
ax[0].label_outer()
imgpcen = librosa.display.specshow(pcen_humpback, x_axis='time', y_axis='mel', ax=ax[1], sr=sampling_rate, hop_length=64)
ax[1].set(title='Per-channel energy normalization (humpback whale)')

fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
fig.colorbar(imgpcen, ax=ax[1])

plt.show()
