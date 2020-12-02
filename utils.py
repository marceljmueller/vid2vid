import torch
from torch import nn
import torch.nn.init as init
from torchvision.utils import make_grid
import os


################################
#### Initialization
################################

def get_gpu_ids():
    '''
    Returns a list of the gpu ids of the machine
    '''
    gpu_ids = []
    try:
        gpu_ids = list(range(0,torch.cuda.device_count()))
        if not isinstance(gpu_ids,list):
            gpu_ids = [gpu_ids]
    except:
        pass
    return gpu_ids

################################
#### GAN Utils
################################

def load_images_from_stitched(image, base_image_size, gpu_ids, testmode = False):
    '''
    Preprocesseses the images and pushes them to the device.

    Parameters:
        image: raw image from dataloader
        base_image_size: int number of pixels for h and width
        gpu_ids: list of gpu_ids or 'cpu'
        testmode: boolean, if True real image is not processed
    '''
    device = torch.device(f'cuda:{gpu_ids[0]}') if len(gpu_ids)>0 else torch.device('cpu')
    image_width = image.shape[3]
    condition = image[:, :, :, :image_width // 2]
    condition = nn.functional.interpolate(condition, size=base_image_size)
    condition = condition.to(device)
    real = None
    if not testmode:
        real = image[:, :, :, image_width // 2:]
        real = nn.functional.interpolate(real, size=base_image_size)
        real = real.to(device)

    return condition, real

def generate_fake_test_mode(generator, dataloader_test, base_image_size, gpu_ids):
    '''
    Generates fake images in without calculating the grads iterating through the dataloader_test.

    Parameters:
        generator: generator
        dataloader_test: torch dataloader for the test set
        base_image_size: int for height/width of the image
    '''
    image, _ = next(iter(dataloader_test))
    cond, real = load_images_from_stitched(image, base_image_size, gpu_ids, testmode=True)

    with torch.no_grad():
        fake_test = generator(cond)
    return fake_test

################################
#### Output Utils
################################

def output_tensorboard(writer, step, args=None, gen_loss=None, disc_loss=None, fake_images=None, fake_images_test=None):
    '''
    Writes training output to Tensorboard.

    Parameters:
        writer: SummaryWriter
        gen_loss: scalar of generator loss
        disc_loss: scalar of generator loss
        step: int current training step
        fake_images: tensor of fake images based on the training set, shape [b, c, h, w|
        fake_images_test: tensor of fake images based on the test set, shape [b, c, h, w|
    '''

    # Write Losse to TB
    if gen_loss is not None:
        writer.add_scalar('Loss/Generator', gen_loss, step)

    if disc_loss is not None:
        writer.add_scalar('Loss/Discriminator', disc_loss, step)

    # Write Images to TB
    if fake_images is not None:
        grid = make_grid(fake_images, nrow=5)
        writer.add_image('fake_images_training_set', grid, args.current_epoch*1000000 + step)

    if fake_images_test is not None:
        grid = make_grid(fake_images_test, nrow=5)
        writer.add_image('fake_images_test_set', grid, args.current_epoch*1000000 + step)

    writer.flush()




