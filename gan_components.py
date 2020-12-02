import torch
from torch import nn
import torch.optim as optim
import torch.nn.init as init
import os
import json
from pdb import set_trace

# Skeleton functions
def channel_map_conv(input_channels, output_channels):
    '''
    Generically transforms t_a with shape [b, c_a, h, w]
    into t_b with shape [b, c_b, h, w], using a kernel with size=1,
    thereby keeping the feature dimensions.

    Parameters:
        input_channels: int number of input channels
        output_channels: int number of output channels
    '''
    return nn.Conv2d(input_channels, output_channels, kernel_size=1)


def conv_layers_unet_block(input_channels, encoder , do_batchnorm=True, dropout_rate=None):
    '''
    Builds a sequence of 2 x Conv2d, Batchnorm (optional), Dropout (optional) plus a final
    Maxpool2d (Encoder only). The first Conv2D doubles (Encoder) or halfs (Decoder) the number of
    input_channels.

    Parameters:
        input_channels: int number of input channels for the first Conv2d
        encoder: boolean, True --> Encoder block, False --> Decoder Block
        do_batchnorm: boolean whether or not to apply Batchnorm
        dropout_rate: float dropout rate
    '''

    # If an encoder block the channel number is doubled , if a decoder block it's halfed
    if encoder:
        inner_number_channels = 2 * input_channels
    else:
        inner_number_channels = input_channels // 2

    # First subset of layers
    stage1_layers = [nn.Conv2d(input_channels, inner_number_channels, kernel_size=3, padding=1)]

    if do_batchnorm:
        stage1_layers.append(nn.BatchNorm2d(inner_number_channels))

    if dropout_rate is not None:
        stage1_layers.append(nn.Dropout(dropout_rate))
    stage1_layers.append(nn.LeakyReLU(0.2))

    # Second subset of layers
    stage2_layers = [nn.Conv2d(inner_number_channels, inner_number_channels, kernel_size=3, padding=1)]

    if do_batchnorm:
        stage2_layers.append(nn.BatchNorm2d(inner_number_channels))

    if dropout_rate is not None:
        stage2_layers.append(nn.Dropout(dropout_rate))
    stage2_layers.append(nn.LeakyReLU(0.2))

    # Final pooling to compress by factor 0.5 (only Encoder)
    if encoder:
        stage2_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*stage1_layers+stage2_layers)


# Initialization functions
def initialize_weights(net, init_method = 'normal'):
    '''
    Initializes the weights of the different layers

    Parameters:
        net: net to be initialized
        init_method: either 'normal' or 'xavier'
    '''
    def init_function(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1):
            if init_method == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
            if init_method == 'xavier':
                 init.xavier_normal_(m.weight.data, gain=0.02)

        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1, 0.02)
            init.constant_(m.bias.data, 0)
    net.apply(init_function)


def initialize_net(net, init_method = 'normal', gpu_ids=[], pretrained = False):
    '''
    Initializes the net. First, it pushes the net to the GPUs (CPU alternatively)
    Second, it initializes the weights. In case the net should be loaded with pretrained
    weights, the weight ini

    Parameters:
        net: network to be initialized
        init_method: 'normal' or 'xavier'
        gpu_ids: int list with the GPUs that should be used by the net, e.g. [0]
        pretrained = boolean, if True the weights are not initialized
    '''
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids) # Need to implement parallel mode.
    if not pretrained:
        initialize_weights(net, init_method)
    return net


# Gradient on/off
def set_requires_grad(net, requires_grad):
    '''
    Sets the requires_grad property for all parameters in the net.

    Parameters:
        net: nn network for which requires_grad should be set
        requires_grad: boolean value
    '''
    for parameter in net.parameters():
        parameter.requires_grad = requires_grad

class EncoderBlock(nn.Module):
    '''
    One block comprises two convolutions optionally including batchnorm and/or dropout,
    which, in total, double the number of input channels, and a final maxpool,
    which compresses height and width by factor 2.

    Parameters:
        input_channels: int number of input_channels
        do_batchnorm: boolean whether or not to apply batchnorm
        dropout_rate: float dropout rate
    '''
    def __init__(self, input_channels, do_batchnorm=True, dropout_rate=0.45):
        super(EncoderBlock,self).__init__()
        self.encoder_block = conv_layers_unet_block(input_channels, True, do_batchnorm, dropout_rate)


    def forward(self, x):
        return self.encoder_block(x)


class DecoderBlock(nn.Module):
    '''
    One block comprises a factor 2 upsampling of height and width, the integration of the skip
    connection output, and three convolutions optionally including batchnorm and/or dropout,
    which, in total, half the number of input channels.

    Parameters:
        input_channels: int number of input_channels
        do_batchnorm: boolean whether or not to apply batchnorm
        dropout_rate: float dropout rate
    '''
    def __init__(self, input_channels, do_batchnorm=True, dropout_rate=None):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')
        self.convolution1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, stride=1, padding=1)
        self.convolution_block = conv_layers_unet_block(input_channels, False, do_batchnorm, dropout_rate)


    def forward(self, x, y_skip_connection):
        x = self.upsample(x)
        x = self.convolution1(x)
        x = torch.cat([x, y_skip_connection], axis = 1)
        x = self.convolution_block(x)

        return x


class Generator(nn.Module):
    '''
    UNet using EncoderBlocks followed by DecoderBlocks as well as
    bounding channel mappings. Skip connection output from the encoder blocks is part of the
    input of the mirrored decoder blocks (see def forward).

    Parameters:
        input_channels: int number of the input channels of the image
        output_channels: int number of the output channels of the image
        dropout_rate: float dropout rate
        number_of_blocks: int number of blocks on the encoder/decoder path respectively
        unet_channels: int number of channels of the first and last layer of the UNET
        lr: float, learning rate of the optimizer (Adam)
    '''
    def __init__(self, input_channels, output_channels, dropout_rate=0.45, number_of_blocks=6, unet_channels=48, lr = 0.0002):
        super(Generator, self).__init__()
        self.channel_map_in = channel_map_conv(input_channels, unet_channels)
        self.channel_multipliers = [(2**i) for i in range(0,number_of_blocks)]
        self.dropout_rates_encoder = [dropout_rate]*number_of_blocks
        self.dropout_rates_decoder = [dropout_rate]*(number_of_blocks-1) + [None]
        self.encoder = nn.ModuleList([
            EncoderBlock(unet_channels*channel_multiplier, dropout_r) for channel_multiplier, dropout_r in zip(self.channel_multipliers, self.dropout_rates_encoder)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(unet_channels*(channel_multiplier*2)) for channel_multiplier, dropout_r in zip(list(reversed(self.channel_multipliers)), self.dropout_rates_decoder)
        ])
        self.channel_map_out = channel_map_conv(unet_channels, output_channels)
        self.optimizer = optim.Adam(self.parameters(), lr = lr)


    def forward(self, x):
        x = self.channel_map_in(x)
        skip_connection_outputs = []
        for encoder_block in self.encoder:
            # Note that the x from the last run of this loop is not
            # appended to self.skip_connection_output
            skip_connection_outputs.append(x)
            x = encoder_block(x)

        for decoder_block, skip_connection_output in zip(self.decoder, reversed(skip_connection_outputs)):
            x = decoder_block(x, skip_connection_output)

        x = self.channel_map_out(x)

        return x


    def backward_generator(self, discriminator, real, condition, fake, gan_criterion, pix_loss_criterion, lambda_pix):
        '''
        Runs the backward pass of the generator and returns the corresponding loss.

        Parameters:
            discriminator: the gan's discriminator
            real: tensor holding the real images
            condition: tensor holding the condition images,
            fake: tensor holding the fake images
            gan_criterion: loss criterion for the gan, typically BCEL
            pix_loss_criterion: pix loss criterion for the generator's loss usually L1
            lambda_pix: weighting of the pix loss criterion in the total cost function
        '''
        self.optimizer.zero_grad()
        disc_out_fake = discriminator(fake, condition)
        gen_loss = gan_criterion(disc_out_fake,torch.ones_like(disc_out_fake)) + lambda_pix *   pix_loss_criterion(fake, real)
        gen_loss.backward()
        self.optimizer.step()
        gen_loss_ret = gen_loss.item()
        return gen_loss_ret


    def initialize(self, gpu_ids, path):
        '''
        Initializes the generator using a pretrained model.

        Parameters:
            gpu_ids = list of gpu_ids
            path = path to the pretrained model
        '''
        # Push to device (parallel mode, therefore has do be done before weight loading)
        self = initialize_net(self, gpu_ids=gpu_ids, pretrained = True)
        # Load the weights from the trained model
        state_dict = torch.load(path)
        self.load_state_dict(state_dict['gen'])


class Discriminator(nn.Module):
    '''
    Patch discriminator built by using an initial channel mapping followed by encoder blocks. The final layer convolves everything to one channel, hence the output is a patch.

    Parameters:
        input_channels: int number of the input channels of the images
        discriminator_channels: int number of the channels of the first encoder block in the discriminator
        number_of_blocks: int number of encoder blocks in the discriminator.

    '''
    def __init__(self, input_channels, output_channels, dropout_rate=0.45, discriminator_channels=32, number_of_blocks=5, lr= 0.0001):
        super(Discriminator, self).__init__()
        self.channel_multipliers = [(2**i) for i in range(0,number_of_blocks)]
        self.do_batchnorm = [False if i==1 else True for i in range(0,number_of_blocks)]
        self.dropout_rate = [dropout_rate] * (number_of_blocks-1) + [None]
        self.channel_map_in = channel_map_conv(input_channels + output_channels, discriminator_channels)
        self.encoder = nn.Sequential(
            *[conv_layers_unet_block(discriminator_channels * channel_multiplier, 1 , do_batchnorm)
            for (channel_multiplier, do_batchnorm) in zip(self.channel_multipliers, self.do_batchnorm)]
        )
        self.patch = nn.Conv2d(2**number_of_blocks * discriminator_channels, 1, kernel_size = 1)
        self.optimizer= optim.Adam(self.parameters(), lr = lr)

    def forward(self, x, y):
        x = torch.cat([x, y], axis = 1)
        x = self.channel_map_in(x)
        x = self.encoder(x)
        x = self.patch(x)
        return x

    def backward_discriminator(self, real, condition, fake, gan_criterion):
        '''
        Runs the backward pass of the discriminator and returns the corresponding loss.

        Parameters:
            real: tensor holding the real images
            condition: tensor holding the condition images,
            fake: tensor holding the fake images
            gan_criterion: loss criterion for the gan, typically BCEL
        '''
        self.optimizer.zero_grad()
        disc_out_fake = self.forward(fake.detach(), condition)
        disc_loss_fake = gan_criterion(disc_out_fake, torch.zeros_like(disc_out_fake))
        disc_out_real = self.forward(real, condition)
        disc_loss_real = gan_criterion(disc_out_real, torch.ones_like(disc_out_real))
        disc_loss = (disc_loss_fake + disc_loss_real) * 0.5
        disc_loss.backward()
        self.optimizer.step()
        disc_loss_ret = disc_loss.item()
        return disc_loss_ret

class GAN():
    '''
    Pix2Pix net consistingof a Generator and a Discriminator as well as the corresponding losses and optimizers.

    Parameters:
        input_channels: int number of the input channels of the image
        output_channels: int number of the output channels of the image
        dropout_rate: float dropout rate
        generator_channels: int number of channels of the first and last layer of the generator
        discriminator_channels: int number of the channels of the first encoder block in the discriminator
        number_blocks_gen: int number of blocks on the encoder/decoder path of the generator respectively
        number_of_blocks_dis: int number of encoder blocks in the discriminator.
        lr_gen: float, learning rate of the generator's optimizer (Adam)
        lr_dis: float, learning rate of the discriminator's optimizer (Adam)
        lambda_pix: weighting of the pix loss criterion in the generator's cost function
    '''
    def __init__(self, input_channels, output_channels, dropout_rate=0.45, generator_channels=48, discriminator_channels=32, number_blocks_gen=6, number_blocks_dis=5, lr_gen=0.0002, lr_dis=0.0001, lambda_pix_loss=350):
        self.generator = Generator(input_channels, output_channels, dropout_rate=dropout_rate, unet_channels=generator_channels, number_of_blocks=number_blocks_gen ,  lr=lr_gen)
        self.discriminator = Discriminator(input_channels=input_channels, output_channels=output_channels, dropout_rate=dropout_rate, discriminator_channels=discriminator_channels, number_of_blocks=number_blocks_dis, lr=lr_dis)
        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.pix_loss_criterion = nn.L1Loss()
        self.lambda_pix_loss = lambda_pix_loss

    def initialize_gan(self, gpu_ids, pretrained_path=None):
        '''
        Initializes the GAN.

        Parameters:
            gpu_ids: int list with the GPUs that should be used by the net, e.g. [0]
            pretrained_path = path of a pretrained model
        '''
        # Load both nets with initial initializer
        if pretrained_path is None:
            self.generator = initialize_net(self.generator, 'normal', gpu_ids)
            self.discriminator = initialize_net(self.discriminator, 'normal', gpu_ids)

        # Load both nets from a previous checkpoint
        elif pretrained_path is not None:
            # Push to device (parallel mode, therefore has do be done before weight loading)
            self.generator = initialize_net(self.generator, gpu_ids=gpu_ids, pretrained = True)
            self.discriminator = initialize_net(self.discriminator, gpu_ids=gpu_ids, pretrained = True)
            # Load the weights from the pretrained model
            state_dict = torch.load(pretrained_path)
            self.generator.load_state_dict(state_dict['gen'])
            self.generator.optimizer.load_state_dict(state_dict['gen_opt'])
            self.discriminator.load_state_dict(state_dict['disc'])
            self.discriminator.optimizer.load_state_dict(state_dict['disc_opt'])

    def training_step(self, condition, real):
        '''
        Runs one training step of the GAN.

        Parameters:
            condition: conditional image
            real: real image
        '''
        fake = self.generator(condition)

        set_requires_grad(self.discriminator, True)
        disc_loss = self.discriminator.backward_discriminator(real, condition, fake, self.gan_criterion)

        set_requires_grad(self.discriminator, False)
        gen_loss = self.generator.backward_generator(self.discriminator, real, condition, fake, self.gan_criterion, self.pix_loss_criterion, self.lambda_pix_loss)
        return disc_loss, gen_loss


    def save_model(self, output_trained_models_path, args, running=False):
        '''
        Saves the model and the config

        Parameters:
            output_trained_models_path: str output path
            args: argpaser.Namespace contains the models parameters
            running: boolean if true the output is writen under *_running.pth and *_running.json, otherwise the filename contains the epoch number, as well (in the later case the output is not overwritten by a later output).
        '''
        if running == False:
            add_save_string = 'e' + str(args.current_epoch)
        else:
            add_save_string = 'running'

        save_path_wo_ext = os.path.join(output_trained_models_path,f'{args.model_name}_{add_save_string}')
        # Save the model
        torch.save({'gen': self.generator.state_dict(),
                            'gen_opt': self.generator.optimizer.state_dict(),
                            'disc': self.discriminator.state_dict(),
                            'disc_opt': self.discriminator.optimizer.state_dict()
                        }, f'{save_path_wo_ext}.pth')

        # Save config file
        with open(f'{save_path_wo_ext}.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        print(f'Model saved after epoch {args.current_epoch} under {save_path_wo_ext}.pth')
