import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

# part1

# 1.1
# creating tones

sampling_frequency = 8192
number_of_samples = 1000
number_of_signals = 10

omega_row = np.array([0.5346, 0.5906, 0.6535, 0.7217])
omega_column = np.array([0.9273, 1.0247, 1.1328])

tone_frequency = np.zeros((number_of_signals, 2))

tone_frequency[0][0] = omega_row[3]
tone_frequency[0][1] = omega_column[1]

index = 1

for i in range(3):
    for j in range(3):
        tone_frequency[index][0] = omega_row[i]
        tone_frequency[index][1] = omega_column[j]
        index += 1

"""
index = 0
for i in range(0, 10, 1):
    print(index)
    print(tone_frequency[index][0])
    print(tone_frequency[index][1])
    index += 1
"""

tone = np.zeros((number_of_signals, number_of_samples))

for i in range(number_of_signals):
    for j in range(number_of_samples):
        low = tone_frequency[i][0]
        high = tone_frequency[i][1]
        tone[i][j] = np.sin(low*j) + np.sin(high*j)

"""
sd.play(tone[0], sampling_frequency)
sd.wait()
"""
# 1.2
# calculate dft

number_of_points = 1024
number_of_tones = 3

tone_part = np.array([tone[0], tone[4], tone[5]])

dft_of_tone_part = np.fft.fft(tone_part, number_of_points, axis=1)

norm_of_dft_of_tone = np.abs(dft_of_tone_part)

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
digit = np.array([0, 4, 5])

"""
for i in range(3):
    axs[i].plot(norm_of_dft_of_tone[i])
    axs[i].set_title(f'digit {digit[i]}')
    axs[i].set_ylabel('Amplitude')

axs[-1].set_xlabel('point')
plt.tight_layout()
plt.show()
"""

# 1.3
# save tone_sequence.wav

identification_number = np.array([0, 3, 1, 2, 1, 1, 1, 3])
zero_padding = np.zeros(100)

id_signal = np.array([])

for digit in identification_number:
    id_signal = np.concatenate((id_signal, tone[digit]))
    id_signal = np.concatenate((id_signal, zero_padding))

audio_id_signal = np.int16(id_signal)
directory = 'part1_audio_files\\tone_sequence.wav'
"""
wavfile.write(directory, sampling_frequency, audio_id_signal)
"""

# 1.4
# calculate fft of windowed signal
