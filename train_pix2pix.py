from gan_components import *
from utils import *

import torch
import torch.optim as optim
import torch.nn.init as init

import argparse

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms

from datetime import datetime
import json

import os
import shutil

from tqdm.auto import tqdm
from pdb import set_trace


if __name__ == '__main__':

    # Parameter-Preprocessing -- Start
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', type=str, help='The run\'s unique name used, e.g., for the storage of checkpoints. Careful, if model_name already exists, the existing checkpoints etc. are overwritten. In case of loading a pretrained model the same model_name can be given as the pretrained model (overwriting is done after model loading).')

    parser.add_argument('-j', '--load_params_from_json_path', type=str, default=None,  help='Path to json file with the parameters for the run. Parameters that are additionally defined via the command line overwrite the values specified in the json file. Do not set this parameter if a pretrained model is used. If a pretrained model is loaded the config file from the pretrained model is automatically loaded, as well.')

    parser.add_argument('-b', '--batch_size', type=int, default=4,  help='Batch size. On a V100 (16GB) 10 is fine.')

    parser.add_argument('-tra', '--training_folder', type=str, default='oliverk', help='Name of the folder with the preprocessed images that should be used for training. Folder has to be a subfolder of ./preprocessed_images. Only the name relative to ./preprocessed_images is necessary.')

    parser.add_argument('-tes', '--test_folder', type=str, default='marcel', help='Name of the folder with the preprocessed images that should be used for testing. Folder has to be a subfolder of ./preprocessed_images. Only the name relative to ./preprocessed_images is necessary.')

    parser.add_argument('-l','--lambda_pix', type=int, default=350, help='Weighting factor of the pix loss in the generator\'s cost function.')

    parser.add_argument('-d','--dropout_rate', type=float, default=0.45, help='Dropout rate used in various layers of the GAN.')

    parser.add_argument('-e','--epochs', type=int, default=30, help='Number of epochs to train the model.')

    parser.add_argument('-cp','--checkpoints', type=int, default=None, nargs='*', help='Sequence of epochs after which the model should be saved. Final model is always saved, even when there is no checkpoint set.')

    parser.add_argument('-os','--output_step', type=int, help='Steps after which intermediate images are sent to tensorboard. These are batch steps, e.g. if batch_size is 8 and output_step is 5 then there is tensorboard output after each 40 processed images. If not set, this is done at the end of each epoch automatically.')

    parser.add_argument('-pt','--pretrained_model_path', type=str, default=None, help='Path to a pretrained model. E.g. ./data/output/prev_runs_name/prev_runs_name_final.pth. If not set the training starts from scratch. If set, all architectural parameters from the save model are used (e.g. ')

    parser.add_argument('-is','--base_image_size', type=int, default=256, help='Image height and width (both conditional and output image).')

    parser.add_argument('-lrg','--learning_rate_generator', type=float, default=0.0001, help='Learning rate for the generator.')

    parser.add_argument('-lrd','--learning_rate_discriminator', type=float, default=0.0002, help='Learning rate for the discriminator.')

    parser.add_argument('-nbbg','--number_of_blocks_generator', type=int, default=6, help='Number of convolutional blocks of one half of the generator. The actual number of blocks is then equal to this number times 2, due to the UNET character.')

    parser.add_argument('-nbbd','--number_of_blocks_discriminator', type=int, default=5, help='Number of convolutional blocks of the discriminator.')

    parser.add_argument('-nbcg','--number_of_channels_generator', type=int, default=48, help='Number of channels of the first convolutional block of the generator.')

    parser.add_argument('-nbcd','--number_of_channels_discriminator', type=int, default=32, help='Number of channels of the first convolutional block of the discriminator.')

    args = parser.parse_args()

    # If a json config file is used the parameters from the file are loaded. Command line parameters overwrite the parameters from the file.
    if args.load_params_from_json_path:
        with open(args.load_params_from_json_path, 'rt') as f:
            temp_args = argparse.Namespace()
            temp_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=temp_args)

    if not args.model_name and args.pretrained_model_path == None:
        raise ValueError('Please set the model_name (-n) or give the path to a pretrained model (-pt *.pth)')

    # If a path for a pretrained model is given the architectural parameters from the pretrained model are used. The json file with the config parameters has to have the same filename (except of the ending) as the pretrained model's .pth file and it has to be within the same folder, as well.
    pretrained_config_path = None
    pretrained_model_path = None
    if not args.load_params_from_json_path and args.pretrained_model_path:
        pretrained_model_path = args.pretrained_model_path
        pretrained_config_path = args.pretrained_model_path.replace('.pth','.json')
        # Set the architectural args equal to None, otherwise they overwrite the files from the config of the pretrained model.
        #for a in ['number_of_channels_discriminator', 'number_of_channels_generator', 'number_of_blocks_discriminator', 'number_of_blocks_generator', 'base_image_size', 'number_of_output_channels', 'number_of_input_channels']:
        #    args.__dict__[a]=None
        with open(pretrained_config_path, 'rt') as f:
            temp_args = argparse.Namespace()
            temp_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=temp_args)

    # If a pretrained is to be loaded args. and args.current_epoch have to be initialized.
    last_epoch_pretrained_model = 0
    if args.pretrained_model_path is None:
        args.__dict__.update({'number_of_images_trained_on': 0})
        args.__dict__.update({'current_epoch': 0})
    # Else, if pretrained model is to be loaded, add the epoch number from the loaded pretrained model so there is there is unique identification later on.
    else:
        last_epoch_pretrained_model = args.current_epoch
        args.epochs = args.epochs + last_epoch_pretrained_model
        print(f'Since the pretrained model has been trained for {last_epoch_pretrained_model} epoch(s),  training now starts at epoch {last_epoch_pretrained_model+1} and goes up to epoch {args.epochs}. Checkpoints are automatically shifted, as well.')

    # Since it is easier to have one dataloader only (conditional image and target (original) image are both extracted from one single stitched together image) the number of input channels and the number of output channels are both fixed to 3. You could change this, (e.g. number_of_input_channels=1 and number_of_output_channels =3) but then you would also have to introduce a separate dataloader for the conditional image and the target image.
    args.__dict__.update({'number_of_input_channels': 3})
    args.__dict__.update({'number_of_output_channels': 3})

    # args is used to store the state of the training process later on in a json file, hence we get rid of some unnecessary info here and add possibly add number_of_training_steps
    args.load_params_from_json_path = None
    args.pretrained_model_path = None

    # Miscellaneous parameter rephrasing
    training_folder = f'./data/preprocessed_images/{args.training_folder}/stitched_frames/'
    test_folder = f'./data/preprocessed_images/{args.test_folder}/stitched_frames/'

    if args.checkpoints is None:
        checkpoints = [args.epochs]
    else:
        checkpoints = [*args.checkpoints]
        # Shift the checkpoints for the same amount as the number of epochs if starting with a pretrained model.
        checkpoints = [c + last_epoch_pretrained_model for c in checkpoints]
        # If the number of the final epoch is not a predefined cp then make it one
        if args.epochs not in checkpoints:
            checkpoints.append(args.epochs)
    # Parameter-Preprocessing -- End


    # Model Initialization -- Start
    # gpu_ids (='cpu' if none)
    gpu_ids = get_gpu_ids()

    gan = GAN(input_channels=args.number_of_input_channels, output_channels=args.number_of_output_channels, dropout_rate=args.dropout_rate, generator_channels=args.number_of_channels_generator, discriminator_channels=args.number_of_channels_discriminator, number_blocks_gen=args.number_of_blocks_generator, number_blocks_dis=args.number_of_blocks_discriminator, lr_gen=args.learning_rate_generator, lr_dis=args.learning_rate_discriminator, lambda_pix_loss=args.lambda_pix)
    gan.initialize_gan(gpu_ids, pretrained_path=pretrained_model_path)

    # Datasets
    transform = transforms.Compose([transforms.Resize((args.base_image_size, args.base_image_size)),transforms.ToTensor()])
    training_set = torchvision.datasets.ImageFolder(training_folder, transform=transform)
    test_set = torchvision.datasets.ImageFolder(test_folder, transform=transform)

    # Summary Writer
    tb_run_label = datetime.now().strftime('%d/%m/%Y-%H:%M') + "_" + args.model_name
    writer = SummaryWriter(os.path.join('./runs',tb_run_label))

    # Make the output folder where the final model and the checkpoints are saved
    output_trained_models_path = f'./data/output/{args.model_name}'
    # If its a new model create (or recreate) the output path.
    if args.number_of_images_trained_on==0:
        if os.path.exists(output_trained_models_path):
            shutil.rmtree(output_trained_models_path)
        os.makedirs(output_trained_models_path, exist_ok= True)
    # Model Initialization -- End

    # Model training -- Start
    dataloader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    step = 0
    fake = None
    fake_test = None
    for epoch in range(1+last_epoch_pretrained_model,args.epochs+1):
        print(f'Epoch {epoch}:')
        args.current_epoch = epoch
        for image_batch , _ in tqdm(dataloader):
            step +=1
            args.number_of_images_trained_on += args.batch_size
            # Load images
            condition, real  = load_images_from_stitched(image_batch, args.base_image_size, gpu_ids)

            # Train GAN for one step
            disc_loss, gen_loss = gan.training_step(condition, real)

            # Put loss to tensorboard
            output_tensorboard(writer, step, gen_loss=gen_loss, disc_loss=disc_loss)

            # Possibly put a set of fakeimages to tensorboard(from the training set and, if defined, from the test set, as well)
            if args.output_step is not None and step % args.output_step == 0:
                fake_test = generate_fake_test_mode(gan.generator, testloader, args.base_image_size, gpu_ids)
                output_tensorboard(writer, step, args=args, gen_loss=None, disc_loss=None, fake_images=fake, fake_images_test=fake_test)

        # Save the model
        if epoch in checkpoints:
                gan.save_model(output_trained_models_path, args)

        # In case no output step is defined image tensorboard output is created at the end of each epoch
        if args.output_step is None:
            fake_test = generate_fake_test_mode(gan.generator, testloader, args.base_image_size, gpu_ids)
            output_tensorboard(writer, step, args=args, gen_loss=None, disc_loss=None, fake_images=fake, fake_images_test=fake_test)

        # Saves model at the end of every fifth epoch (overwrites the one from the epoch before). Only if no checkpoint for this epoch.
        if epoch not in checkpoints and epoch % 5 == 0:
            gan.save_model(output_trained_models_path, args, running=True)

    writer.close()
    # Model training -- End



