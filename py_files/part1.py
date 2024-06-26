import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import find_peaks

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


tone = np.zeros((number_of_signals, number_of_samples))

for i in range(number_of_signals):
    low = tone_frequency[i][0]
    high = tone_frequency[i][1]
    for j in range(number_of_samples):
        tone[i][j] = np.sin(low*j) + np.sin(high*j)


sd.play(tone[0], sampling_frequency)
sd.wait()

# 1.2
# calculate dft

number_of_points = 1024
number_of_tones = 3

tone_part = np.array([tone[0], tone[4], tone[5]])

dft_of_tone_part = np.fft.fft(tone_part, number_of_points, axis=1)

norm_of_dft_of_tone = np.abs(dft_of_tone_part)

digit = np.array([0, 7, 9])


fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

for i in range(3):
    axs[i].plot(norm_of_dft_of_tone[i])
    axs[i].set_title(f'digit {digit[i]}')
    axs[i].set_ylabel('Amplitude')

axs[-1].set_xlabel('point')
plt.tight_layout()
plt.show()


# 1.3
# save tone_sequence.wav

identification_number = np.array([0, 3, 1, 2, 1, 1, 1, 3])
zero_padding = np.zeros(100)

id_signal = np.array([])

for digit in identification_number:
    id_signal = np.concatenate((id_signal, tone[digit]))
    id_signal = np.concatenate((id_signal, zero_padding))

audio_id_signal = np.int16(id_signal)

directory = '..\\part1_audio_files\\tone_sequence.wav'


wavfile.write(directory, sampling_frequency, audio_id_signal)


# 1.4
# calculate fft of windowed signals

window_length = 1000
rect_window = np.ones(window_length)
hamming_window = np.hamming(window_length)

id_size = np.size(identification_number)
windowed_signal_rect = np.zeros((id_size, window_length))
windowed_signal_hamming = np.zeros((id_size, window_length))

index = 0
for i in range(id_size):
    part = id_signal[index:index+window_length]
    windowed_signal_rect[i] = np.multiply(rect_window, part)
    windowed_signal_hamming[i] = np.multiply(hamming_window, part)
    index = index + window_length + np.size(zero_padding)

points = 1024
dft_of_rect = np.fft.fft(windowed_signal_rect, points, axis=1)
dft_of_hamming = np.fft.fft(windowed_signal_hamming, points, axis=1)


index = 3


plt.plot(np.abs(dft_of_rect[index]))
plt.xlabel('dft points')
plt.ylabel('amplitude')
plt.title(f'digit {identification_number[index]}')
plt.show()

plt.plot(np.abs(dft_of_hamming[index]))
plt.xlabel('dft points')
plt.ylabel('amplitude')
plt.title(f'digit {identification_number[index]}')
plt.show()


# 1.5
# 

# 1.6
# write function that decodes a signal
# it returns a vector - the digits of the decoded signal

def frequencies_to_digit(frequencies, table):
    number_of_digits = 10
    distance = np.zeros(number_of_digits)

    for i in range(number_of_digits):
        distance[i] = np.linalg.norm(frequencies-table[i])

    mathcing_digit = np.argmin(distance)
    return mathcing_digit

def ttdecode(encoded_signal):
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

    window_length = 1000
    padding_length = 100
    tone_length = window_length + padding_length
    number_of_digits = round(np.size(encoded_signal) / tone_length)
    points = 1024
    hamming_window = np.hamming(window_length)
    windowed_signal = np.zeros((number_of_digits, window_length))

    index = 0
    for i in range(number_of_digits):
        part = encoded_signal[index:index+window_length]
        windowed_signal[i] = np.multiply(hamming_window, part)
        index = index + tone_length

    dft_of_signal = np.fft.fft(windowed_signal, points, axis=1)
    amplitude_of_dft = np.abs(dft_of_signal)

    peaks = np.zeros((number_of_digits, 2))

    for i in range(number_of_digits):
        amp = amplitude_of_dft[i][0:int((points/2))]
        peaks[i], _ = find_peaks(amp, height=50, distance=10)

    constant = 2 * np.pi / points
    peaks = peaks * constant
    
    digits = np.zeros(number_of_digits)
    for i in range(number_of_digits):
        digits[i] = frequencies_to_digit(peaks[i], tone_frequency)
    
    return digits

decoded_signal = ttdecode(id_signal)

for digit in decoded_signal:
    print(int(digit), end=" ")

print("")


# 1.7
# load files and decode them

easy_sig = np.load("..\\data\\easy_sig.npy")
medium_sig = np.load("..\\data\\medium_sig.npy")
hard_sig = np.load("..\\data\\hard_sig.npy")

def print_decoded(signal):
    decoded_signal = ttdecode(signal)
    for digit in decoded_signal:
        print(int(digit), end=" ")
    
    print("")


print("easy_sig", end=": ")
print_decoded(easy_sig)

print("medium_sig", end=": ")
print_decoded(medium_sig)

print("hard_sig", end=": ")
print_decoded(hard_sig)




