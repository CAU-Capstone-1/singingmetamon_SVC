import numpy as np
import librosa
import scipy.io.wavfile

audio_path=''
save_path=''
time=0
sample_rate=16000

#무음제거 및 타임커팅
def time_cutting(time, audio_path, save_path,resample):
    audio, sr = librosa.load(audio_path)
    non_silence_indices = librosa.effects.split(audio, top_db=30)
    audio = np.concatenate([audio[start:end] for start, end in non_silence_indices])
    real_audio = librosa.resample(audio,sr,resample)
    j=1
    for i in range(int(len(real_audio)/(resample*time))):
        temp_audio = real_audio[resample*time*i:resample*time*(i+1)]
        scipy.io.wavfile.write(save_path+str(j)+ '.wav' ,resample, temp_audio)
        j=j+1
    print('....split 파일 생성 완료....')

time_cutting(time, audio_path, save_path, sample_rate)