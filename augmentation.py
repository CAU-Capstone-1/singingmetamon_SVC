import numpy as np
import librosa
import soundfile as sf

def adding_white_noise(data,i, sr=16000, noise_rate=0.005,save_path='.'):
    wn = np.random.randn(len(data))
    print(wn)
    data_wn = data + noise_rate*wn
    sf.write(save_path+'wn'+str(i)+'.wav', data_wn, samplerate=sr,format='WAV', endian='LITTLE', subtype='PCM_16')
    print('White Noise 저장 성공')

    return data_wn


def shifting_sound(data, i, sr=16000, roll_rate=0.3, save_path='.'):
    data_roll = np.roll(data, int(len(data) * roll_rate))
    sf.write(save_path + 'roll' + str(i) + '.wav', data_roll, samplerate=sr, format='WAV', endian='LITTLE',
             subtype='PCM_16')
    print('rolling_sound 저장 성공')

    return data_roll


def minus_sound(data,i, sr=16000,save_path='.'):
    data_mn = (-1)*data
    sf.write(save_path+'mn'+str(i)+'.wav', data_mn ,samplerate=sr, format='WAV', endian='LITTLE', subtype='PCM_16')
    print('minus_data 저장 성공')

    return data_mn

load_path = ''
save_path = ''
splited_audios = 0

for i in range(splited_audios):
    try:
        data, sr = librosa.load(load_path + str(i) + 'wav', sr=16000)
        save_path = save_path
        adding_white_noise(data, i, save_path=save_path)
        shifting_sound(data, i, save_path=save_path)
        minus_sound(data, i, save_path=save_path)
    except FileNotFoundError:
        continue

