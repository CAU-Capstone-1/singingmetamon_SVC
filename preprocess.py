import librosa
import numpy as np
import os
import pyworld
from librosa.core import stft


def get_wave_files(audio_path, sampling_rate):
    results = []
    for file in os.listdir(audio_path):
        file_path = os.path.join(audio_path, file)
        wav, sr = librosa.load(file_path, sr=sampling_rate, mono=True)
        results.append(wav)

    return results


def transpose_list(arr):
    transposed = []
    for array in arr:
        transposed.append(array.T)
    return transposed


def logf0_info(fzero):
    fzero = np.ma.log(np.concatenate(fzero))
    mean = fzero.mean()
    std = fzero.std()

    return mean,std


def fzero_converter(fzero, mean_source, std_source, mean_target, std_target):
    result = np.exp((np.log(fzero) - mean_source) / std_source * std_target + mean_target)

    return result


def padding_wav(wav, sampling_rate, frame_period, multiple=4):
    assert wav.ndim == 1
    frames_num = len(wav)
    num_frames_padded = int(
        (np.ceil((np.floor(frames_num / (sampling_rate * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (
                    sampling_rate * frame_period / 1000))
    num_frames_diff = num_frames_padded - frames_num
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values=0)

    return wav_padded


def sampling_data(dataset_A, dataset_B, n_frames=512):
    num_samples = min(len(dataset_A), len(dataset_B))
    A_idxs = np.arange(len(dataset_A))
    B_idxs = np.arange(len(dataset_B))
    np.random.shuffle(A_idxs)
    np.random.shuffle(B_idxs)
    A_idx_subset = A_idxs[:num_samples]
    B_idx_subset = A_idxs[:num_samples]

    train_data_A = []
    train_data_B = []

    for idx_A, idx_B in zip(A_idx_subset, B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        if frames_A_total < n_frames:
            for i in range(num_samples):
                idx_A = idx_A + 1
                data_A = dataset_A[idx_A]
                frames_A_total = data_A.shape[1]
                if frames_A_total > n_frames:
                    start_A = np.random.randint(frames_A_total - n_frames + 1)
                    end_A = start_A + n_frames
                    train_data_A.append(data_A[:, start_A:end_A])
                    break
        else:
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            train_data_A.append(data_A[:, start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]

        if frames_B_total < n_frames:
            for j in range(num_samples):
                idx_B = idx_B + 1
                data_B = dataset_B[idx_B]
                frames_B_total = data_B.shape[1]
                if frames_B_total > n_frames:
                    start_B = np.random.randint(frames_B_total - n_frames + 1)
                    end_B = start_B + n_frames
                    train_data_B.append(data_B[:, start_B:end_B])
                    break
        else:
            start_B = np.random.randint(frames_B_total - n_frames + 1)
            end_B = start_B + n_frames
            train_data_B.append(data_B[:, start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B



def world_decompose(wav, fs, frame_period=5.0):
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    return f0, timeaxis, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=24):
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp


def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp


def world_encode_data(wavs, fs, frame_period=5.0, coded_dim=24):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()

    for i in range(len(wavs)):
        f0, timeaxis, sp, ap = world_decompose(wav=wavs[i], fs=fs, frame_period=frame_period)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)
        print('{}_th file encoding ...'.format(i))

    print('Encoding is done')
    return f0s, timeaxes, sps, aps, coded_sps




def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    wav = wav.astype(np.float32)

    return wav


def coded_sps_normalization_fit_transform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized, coded_sps_mean, coded_sps_std

