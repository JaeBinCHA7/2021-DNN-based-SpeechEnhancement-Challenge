"""
Where the model is actually trained and validated
"""

import torch
import numpy as np
import tools_for_model as tools
import config as cfg
from tools_for_estimate import cal_pesq, cal_stoi

L1Loss = torch.nn.L1Loss()


#######################################################################
#                             For train                               #
#######################################################################
def model_train(model, optimizer, train_loader, direct, DEVICE):
    # initialization
    train_loss = 0
    train_main_loss = 0
    train_perceptual_loss = 0
    batch_num = 0

    # train
    model.train()
    if cfg.perceptual != 'False':
        for inputs, targets in tools.Bar(train_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outspec, outputs = model(inputs, direct_mapping=direct)
            main_loss = model.loss(outputs, targets)
            perceptual_loss = model.loss(outputs, targets, outspec, perceptual=True)

            # the constraint ratio
            r1 = 1
            r2 = 1
            r3 = r1 + r2
            loss = (r1 * main_loss + r2 * perceptual_loss) / r3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_main_loss += main_loss
            train_perceptual_loss += perceptual_loss
        train_loss /= batch_num
        train_main_loss /= batch_num
        train_perceptual_loss /= batch_num

        return train_loss, train_main_loss, train_perceptual_loss
    else:
        for inputs, targets in tools.Bar(train_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            _, outputs = model(inputs, direct_mapping=direct)

            loss = model.loss(outputs, targets)
            # # if you want to check the scale of the loss
            # print('loss: {:.4}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
        train_loss /= batch_num

        return train_loss


def cycle_model_train(N2C, C2N, optimizer, train_loader, direct, DEVICE):
    # initialization
    train_loss = 0
    train_main_loss = 0
    train_C2N_NL1_loss = 0
    train_N2C_CL1_loss = 0
    batch_num = 0

    N2C.train()
    C2N.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        _, estimated_clean_outputs = N2C(inputs, direct_mapping=direct)
        _, fake_noisy_outputs = C2N(estimated_clean_outputs, direct_mapping=True)

        _, estimated_noisy_outputs = C2N(targets, direct_mapping=True)
        _, fake_clean_outputs = N2C(estimated_noisy_outputs, direct_mapping=direct)

        main_loss = N2C.loss(estimated_clean_outputs, targets)

        C2N_NL1_loss = L1Loss(fake_noisy_outputs, inputs)
        N2C_CL1_loss = L1Loss(fake_clean_outputs, targets)

        # constraint ratio
        r1 = 1
        r2 = 10000
        r3 = 5000
        r = r1 + r2 + r3

        # # if you want to check the scale of the loss
        # print('M: {:.6} C2N: {:.6} N2C {:.6}'.format(main_loss, C2N_NL1_loss, N2C_CL1_loss))
        loss = (r1 * main_loss + r2 * C2N_NL1_loss + r3 * N2C_CL1_loss) / r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        train_main_loss += main_loss
        train_C2N_NL1_loss += C2N_NL1_loss
        train_N2C_CL1_loss += N2C_CL1_loss
    train_loss /= batch_num
    train_main_loss /= batch_num
    train_C2N_NL1_loss /= batch_num
    train_N2C_CL1_loss /= batch_num

    return train_loss, train_main_loss, train_C2N_NL1_loss, train_N2C_CL1_loss


#######################################################################
#                           For validation                            #
#######################################################################
def model_validate(model, validation_loader, direct, writer, epoch, DEVICE):
    # initialization
    validation_loss = 0
    validation_main_loss = 0
    validation_perceptual_loss = 0
    batch_num = 0

    all_batch_input = []
    all_batch_target = []
    all_batch_output = []

    # save the same sample
    clip_num = 10

    model.eval()
    with torch.no_grad():
        if cfg.perceptual != 'False':
            for inputs, targets in tools.Bar(validation_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                outspec, outputs = model(inputs, direct_mapping=direct)
                main_loss = model.loss(outputs, targets)
                perceptual_loss = model.loss(outputs, targets, outspec, perceptual=True)

                # the constraint ratio
                r1 = 1
                r2 = 1
                r3 = r1 + r2
                loss = (r1 * main_loss + r2 * perceptual_loss) / r3

                validation_loss += loss
                validation_main_loss += main_loss
                validation_perceptual_loss += perceptual_loss

                # for saving the sample we want to tensorboard
                if epoch % 10 == 0:
                    # all batch data array
                    all_batch_input.extend(inputs)
                    all_batch_target.extend(targets)
                    all_batch_output.extend(outputs)

            # save the samples to tensorboard
            if epoch % 10 == 0:
                writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                            all_batch_target[clip_num],
                                            all_batch_output[clip_num], epoch)
            validation_loss /= batch_num
            validation_main_loss /= batch_num
            validation_perceptual_loss /= batch_num

            return validation_loss, validation_main_loss, validation_perceptual_loss
        else:
            for inputs, targets in tools.Bar(validation_loader):
                batch_num += 1

                # to cuda
                inputs = inputs.float().to(DEVICE)
                targets = targets.float().to(DEVICE)

                _, outputs = model(inputs, direct_mapping=direct)
                loss = model.loss(outputs, targets)

                validation_loss += loss

                # for saving the sample we want to tensorboard
                if epoch % 10 == 0:
                    # all batch data array
                    all_batch_input.extend(inputs)
                    all_batch_target.extend(targets)
                    all_batch_output.extend(outputs)

            # save the samples to tensorboard
            if epoch % 10 == 0:
                writer.save_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                            all_batch_target[clip_num],
                                            all_batch_output[clip_num], epoch)

            validation_loss /= batch_num

            return validation_loss


def cycle_model_validate(N2C, C2N, validation_loader, direct, writer, epoch, DEVICE):
    # initialization
    validation_loss = 0
    validation_main_loss = 0
    validation_C2N_NL1_loss = 0
    validation_N2C_CL1_loss = 0
    batch_num = 0

    all_batch_input = []
    all_batch_target = []
    all_batch_clean_output = []
    all_batch_noisy_output = []

    # save the same sample
    clip_num = 10

    N2C.eval()
    C2N.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(validation_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            _, estimated_clean_outputs = N2C(inputs, direct_mapping=direct)
            _, fake_noisy_outputs = C2N(estimated_clean_outputs, direct_mapping=True)

            _, estimated_noisy_outputs = C2N(targets, direct_mapping=True)
            _, fake_clean_outputs = N2C(estimated_noisy_outputs, direct_mapping=direct)

            main_loss = N2C.loss(estimated_clean_outputs, targets)

            C2N_NL1_loss = L1Loss(fake_noisy_outputs, inputs)
            N2C_CL1_loss = L1Loss(fake_clean_outputs, targets)

            # constraint ratio
            r1 = 1
            r2 = 10000
            r3 = 5000
            r = r1 + r2 + r3

            loss = (r1 * main_loss + r2 * C2N_NL1_loss + r3 * N2C_CL1_loss) / r

            # for saving the sample we want to tensorboard
            if epoch % 10 == 0:
                # all batch data array
                all_batch_input.extend(inputs)
                all_batch_target.extend(targets)
                all_batch_clean_output.extend(estimated_clean_outputs)
                all_batch_noisy_output.extend(estimated_noisy_outputs)

            validation_loss += loss
            validation_main_loss += main_loss
            validation_C2N_NL1_loss += C2N_NL1_loss
            validation_N2C_CL1_loss += N2C_CL1_loss
        validation_loss /= batch_num
        validation_main_loss /= batch_num
        validation_C2N_NL1_loss /= batch_num
        validation_N2C_CL1_loss /= batch_num

        # save the samples to tensorboard
        if epoch % 10 == 0:
            writer.save_cycle_samples_we_want('clip: ' + str(clip_num), all_batch_input[clip_num],
                                              all_batch_target[clip_num], all_batch_clean_output[clip_num],
                                              all_batch_noisy_output[clip_num], epoch)
    return validation_loss, validation_main_loss, validation_C2N_NL1_loss, validation_N2C_CL1_loss


#######################################################################
#                           For evaluation                            #
#######################################################################
def model_eval(model, validation_loader, direct, dir_to_save, epoch, DEVICE):
    # initialize
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')

    model.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(validation_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            _, outputs = model(inputs, direct_mapping=direct)

            # estimate the output speech with pesq and stoi
            estimated_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = cal_pesq(estimated_wavs, clean_wavs)
            stoi = cal_stoi(estimated_wavs, clean_wavs)

            # pesq: 0.1 better / stoi: 0.01 better
            for i in range(len(pesq)):
                f_score.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq[i], stoi[i]))

            # reshape for sum
            pesq = np.reshape(pesq, (1, -1))
            stoi = np.reshape(stoi, (1, -1))

            avg_pesq += sum(pesq[0]) / len(inputs)
            avg_stoi += sum(stoi[0]) / len(inputs)
        avg_pesq /= batch_num
        avg_stoi /= batch_num
    f_score.close()
    return avg_pesq, avg_stoi
