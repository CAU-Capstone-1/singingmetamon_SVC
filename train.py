import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import argparse
from model import CycleGAN
import extracting_feature
from preprocess import *


def store_to_file(doc, file_name):
    doc = doc + "\n"
    with open(file_name, "a") as myfile:
        myfile.write(doc)

def train(logf0s_normalization,
          mcep_normalization,
          coded_sps_A_norm,
          coded_sps_B_norm,
          model_checkpoint_dir,
          model_checkpoint_file,
          validation_A_dir,
          validation_B_dir,
          output_A_dir,
          output_B_dir,
          log_dir,
          tensorboard_dir,
          restart_training_at=None):

    file_name = log_dir

    start_epoch = 0
    num_epochs = 3000
    mini_batch_size = 1

    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 200000
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / 200000
    lambda_cycle = 10
    lambda_identity = 5
    sr = 16000
    n_features = 24
    frame_period = 5.0
    n_frames = 512
    num_mcep = 24

    gamma_A = 0.5
    gamma_B = 0.5
    lambda_k_A = 0.001
    lambda_k_B = 0.001
    balance_A = 0
    balance_B = 0
    # kta 초기값
    k_t_A = 0
    k_t_B = 0

    generator_lr = 0.0002
    generator_lr_decay = generator_lr / 200000
    discriminator_lr = 0.0001
    discriminator_lr_decay = discriminator_lr / 200000
    cycle_lambda = 10
    identity_lambda = 5

    coded_sps_A_norm = extracting_feature.load_pickle_file(coded_sps_A_norm)
    coded_sps_B_norm = extracting_feature.load_pickle_file(coded_sps_B_norm)

    logf0s_normalization = np.load(logf0s_normalization)
    log_f0s_mean_A = logf0s_normalization['mean_A']
    log_f0s_std_A = logf0s_normalization['std_A']
    log_f0s_mean_B = logf0s_normalization['mean_B']
    log_f0s_std_B = logf0s_normalization['std_B']

    mcep_normalization = np.load(mcep_normalization)
    coded_sps_A_mean = mcep_normalization['mean_A']
    coded_sps_A_std = mcep_normalization['std_A']
    coded_sps_B_mean = mcep_normalization['mean_B']
    coded_sps_B_std = mcep_normalization['std_B']


    if os.path.exists(tensorboard_dir) is False :
        os.mkdir(tensorboard_dir)
    if validation_A_dir is not None:
        if not os.path.exists(output_A_dir):
            os.makedirs(output_A_dir)

    if validation_B_dir is not None:
        if not os.path.exists(output_B_dir):
            os.makedirs(output_B_dir)

    model = CycleGAN(num_features = n_features, log_dir = tensorboard_dir)

    if restart_training_at is not None:
        start_epoch = model.load(restart_training_at)
        print("Training resumed")

        
    print("Training start")
    for epoch in range(start_epoch+1, num_epochs+1) :
        print("Epoch : %d " % epoch )
        start_time = time.time()
        train_A, train_B = sampling_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)

        n_samples = train_A.shape[0]
        sentence = "Epoch: {}".format(epoch)
        store_to_file(sentence, file_name)
        for i in range(n_samples) :
            n_iter = (n_samples * (epoch-1)) + i
            if n_iter % 50 == 0:
                k_t_A = k_t_A + (lambda_k_A *balance_A)
                if k_t_A > 1:
                    k_t_A = 1
                if k_t_A < 0 :
                    k_t_A = 0
                k_t_B = k_t_B + (lambda_k_B *balance_B)
                if k_t_B > 1.0:
                    k_t_B = 1.0
                if k_t_B < 0. :
                    k_t_B = 0.
            if n_iter > 10000 :
                identity_lambda = 0
            if n_iter > 200000 :
                generator_lr = max(0, generator_lr - generator_lr_decay)
                discriminator_lr = max(0, discriminator_lr - discriminator_lr_decay)
            start = i
            end = start + 1

            generator_loss, discriminator_loss, measure_A, measure_B, k_t_A, k_t_B, balance_A, balance_B = model.train(
                            input_A=train_A[start:end], input_B=train_B[start:end],
                            lambda_cycle=lambda_cycle,
                            lambda_identity=lambda_identity,
                            gamma_A=gamma_A, gamma_B=gamma_B, lambda_k_A=lambda_k_A, lambda_k_B=lambda_k_B,
                            generator_learning_rate=generator_learning_rate,
                            discriminator_learning_rate=discriminator_learning_rate,
                            k_t_A = k_t_A, k_t_B = k_t_B)
            if  n_iter % 50 == 0:
                sentence = 'Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(
                    n_iter, generator_learning_rate, discriminator_learning_rate, generator_loss,
                    discriminator_loss)
                store_to_file(sentence, file_name)
                print(
                    'Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(
                        n_iter, generator_learning_rate, discriminator_learning_rate, generator_loss,
                        discriminator_loss))
                print(
                    'Measure_A: {:.3f}, measure_B: {:.3f}, k_t_A: {:.3f}, k_t_B: {:.3f}'.format(measure_A,
                                                                                                measure_B,
                                                                                                k_t_A,
                                                                                                k_t_B))
        end_time = time.time()
        epoch_time = end_time-start_time
        print("Generator Loss : %f, Discriminator Loss : %f, Time : %02d:%02d:%02d" % (generator_loss, discriminator_loss,(epoch_time % 3600 // 60),(epoch_time % 60 // 1), (epoch_time % 60 // 1)))

        if epoch % 50 == 0:
            print("Saving model Checkpoint")
            sentence = "Saving model Checkpoint"
            store_to_file(sentence, file_name)
            model.save(directory = model_checkpoint_dir, filename = model_checkpoint_file, epoch=epoch)
            print("Model Saved!")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Train CycleGAN model for datasets.')

    logf0s_normalization_default = '../cache/logf0s_normalization.npz'
    mcep_normalization_default = '../cache/mcep_normalization.npz'
    coded_sps_A_norm_default = '../cache/coded_sps_A_norm.pickle'
    coded_sps_B_norm_default = '../cache/coded_sps_B_norm.pickle'
    model_checkpoint_dir_default = './checkpoint/default'
    model_checkpoint_file_default = 'default.ckpt'
    validation_A_dir_default = './data/test_default/A'
    validation_B_dir_default = './data/test_default/B'
    output_A_dir_default = './validation_output/A'
    output_B_dir_default = './validation_output/B'
    log_dir_defalut = './log/default'
    tensorboard_dir_defalut = './tensorboard/default'
    resume_training_at_default = None
    cuda_default = None

    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--coded_sps_A_norm', type=str,
                        help="mcep norm for data A", default=coded_sps_A_norm_default)
    parser.add_argument('--coded_sps_B_norm', type=str,
                        help="mcep norm for data B", default=coded_sps_B_norm_default)
    parser.add_argument('--model_checkpoint_dir', type=str, help='Directory for saving models.', default=model_checkpoint_dir_default)
    parser.add_argument('--model_checkpoint_file', type=str, help='File name for saving model.', default=model_checkpoint_file_default)
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
    parser.add_argument('--log_dir', type=str,
                        help="log_file while training", default=log_dir_defalut)
    parser.add_argument('--tensorboard_dir', type=str, default=tensorboard_dir_defalut)
    parser.add_argument('--resume_training_at', type=str,
                        help="Location of the pre-trained model to resume training",
                        default=resume_training_at_default)
    parser.add_argument('--cuda', type=str, default=cuda_default)

    argv = parser.parse_args()

    logf0s_normalization = argv.logf0s_normalization
    mcep_normalization = argv.mcep_normalization
    coded_sps_A_norm = argv.coded_sps_A_norm
    coded_sps_B_norm = argv.coded_sps_B_norm
    model_checkpoint_dir = argv.model_checkpoint_dir
    model_checkpoint_file = argv.model_checkpoint_file
    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_A_dir = argv.output_A_dir
    output_B_dir = argv.output_B_dir
    log_dir = argv.log_dir
    tensorboard_dir = argv.tensorboard_dir
    resume_training_at = argv.resume_training_at
    cuda = argv.cuda

    if not os.path.exists(logf0s_normalization) or not os.path.exists(mcep_normalization):
        print("Preprocessed data does not exist. Please pre-processing first.")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda


    train(logf0s_normalization=logf0s_normalization,
          mcep_normalization=mcep_normalization,
          coded_sps_A_norm=coded_sps_A_norm,
          coded_sps_B_norm=coded_sps_B_norm,
          model_checkpoint_dir=model_checkpoint_dir,
          model_checkpoint_file =model_checkpoint_file,
          validation_A_dir=validation_A_dir,
          validation_B_dir=validation_B_dir,
          output_A_dir=output_A_dir,
          output_B_dir=output_B_dir,
          log_dir=log_dir,
          tensorboard_dir=tensorboard_dir,
          restart_training_at=resume_training_at)

    print("Training Done!")
