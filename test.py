import argparse
import soundfile as sf
import extracting_feature
from preprocess import *
from model import CycleGAN

def test(logf0s_normalization,
          mcep_normalization,
          model_checkpoint_dir,
          validation_A_dir,
          validation_B_dir,
          output_A_dir,
          output_B_dir):

    sr = 16000
    n_features = 24
    frame_period = 5.0
    num_mcep = 24

    if validation_A_dir is not None:
        if not os.path.exists(output_A_dir):
            os.makedirs(output_A_dir)

    if validation_B_dir is not None:
        if not os.path.exists(output_B_dir):
            os.makedirs(output_B_dir)

    model = CycleGAN(num_features=n_features, mode="test")
    model.load(model_checkpoint_dir)

    logf0s_normalization = np.load(logf0s_normalization)
    log_f0s_mean_A = logf0s_normalization['A_mean']
    log_f0s_std_A = logf0s_normalization['A_std']
    log_f0s_mean_B = logf0s_normalization['B_mean']
    log_f0s_std_B = logf0s_normalization['B_std']

    mcep_normalization = np.load(mcep_normalization)
    coded_sps_A_mean = mcep_normalization['A_mean']
    coded_sps_A_std = mcep_normalization['A_std']
    coded_sps_B_mean = mcep_normalization['B_mean']
    coded_sps_B_std = mcep_normalization['B_std']

    if validation_A_dir is not None:
        print('Converting Data A into Voice B...')
        for file in os.listdir(validation_A_dir):
            filepath = os.path.join(validation_A_dir, file)
            try:
                wav, _ = librosa.load(filepath, sr=sr)
                wav = padding_wav(wav=wav, sr=sr, frame_period=frame_period, multiple=4)
                f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sr, frame_period=frame_period)
                f0_converted = fzero_converter(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                                mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
                coded_sp = world_encode_spectral_envelop(sp=sp, fs=sr, dim=num_mcep)
                coded_sp_transposed = coded_sp.T
                coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='A2B')[0]
                coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                coded_sp_converted = coded_sp_converted.T
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted,
                                                                     fs=sr)
                wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted,
                                                         ap=ap, fs=sr, frame_period=frame_period)
                temp = os.path.join(output_A_dir, os.path.basename(file))
                sf.write(temp, wav_transformed, sr)
            except Exception as e:
                print(validation_A_dir, e)
                continue

    if validation_B_dir is not None:
        print('Converting Data B into Voice A...')
        for file in os.listdir(validation_B_dir):
            try:
                filepath = os.path.join(validation_B_dir, file)
                wav, _ = librosa.load(filepath, sr=sr)
                wav = padding_wav(wav=wav, sr=sr, frame_period=frame_period, multiple=4)
                f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sr, frame_period=frame_period)
                f0_converted = fzero_converter(f0=f0, mean_log_src=log_f0s_mean_B, std_log_src=log_f0s_std_B,
                                                mean_log_target=log_f0s_mean_A, std_log_target=log_f0s_std_A)
                coded_sp = world_encode_spectral_envelop(sp=sp, fs=sr, dim=num_mcep)
                coded_sp_transposed = coded_sp.T
                coded_sp_norm = (coded_sp_transposed - coded_sps_B_mean) / coded_sps_B_std
                coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='B2A')[0]
                coded_sp_converted = coded_sp_converted_norm * coded_sps_A_std + coded_sps_A_mean
                coded_sp_converted = coded_sp_converted.T
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted,
                                                                     fs=sr)
                wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted,
                                                         ap=ap, fs=sr, frame_period=frame_period)
                temp = os.path.join(output_B_dir, os.path.basename(file))
                sf.write(temp, wav_transformed, sr)
            except:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CycleGAN model for datasets.')

    logf0s_normalization_default = '../cache/logf0s_normalization.npz'
    mcep_normalization_default = '../cache/mcep_normalization.npz'
    model_checkpoint_dir_default = './checkpoint/default'
    validation_A_dir_default = './data/test_default/A'
    validation_B_dir_default = './data/test_default/B'
    output_A_dir_default = './validation_output/A'
    output_B_dir_default = './validation_output/B'
    cuda_default = None

    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--model_checkpoint_dir', type=str, help='Directory for saving models.',
                        default=model_checkpoint_dir_default)
    parser.add_argument('--validation_A_dir', type=str,
                        help='Convert validation A after each training epoch. If set none, no conversion would be done during the training.',
                        default=validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str,
                        help='Convert validation B after each training epoch. If set none, no conversion would be done during the training.',
                        default=validation_B_dir_default)
    parser.add_argument('--output_A_dir', type=str, help='output for converted Sound Source A',
                        default=output_A_dir_default)
    parser.add_argument('--output_B_dir', type=str, help='output for converted Sound Source B',
                        default=output_B_dir_default)
    parser.add_argument('--cuda', type=str, default=cuda_default)

    argv = parser.parse_args()

    logf0s_normalization = argv.logf0s_normalization
    mcep_normalization = argv.mcep_normalization
    model_checkpoint_dir = argv.model_checkpoint_dir
    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_A_dir = argv.output_A_dir
    output_B_dir = argv.output_B_dir
    cuda = argv.cuda

    if not os.path.exists(logf0s_normalization) or not os.path.exists(mcep_normalization):
        print("Preprocessed data does not exist. Please pre-processing first.")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    test(logf0s_normalization=logf0s_normalization,
          mcep_normalization=mcep_normalization,
          model_checkpoint_dir=model_checkpoint_dir,
          validation_A_dir=validation_A_dir,
          validation_B_dir=validation_B_dir,
          output_A_dir=output_A_dir,
          output_B_dir=output_B_dir)
    print("Test.py Sample Create Completed!")
