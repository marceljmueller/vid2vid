from gan_components import *
from utils import *

import argparse

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import imageio
import moviepy.editor as mp
import math

import os
import shutil
from tqdm.auto import tqdm

if __name__ =='__main__':
    # Parameter Preprocessing -- Start
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder', type=str, help='Name of the folder with the input images. This has to be a folder directly under ./data/preprocessed_images. The program then automatically looks for stitched_frames/all in this folder. If not set the program looks if a test_folder was assigned in the trained model. If so, it uses this folder as input_foler.')

    parser.add_argument('-t', '--trained_model_path', type=str, required=True,  help='Full path to the trained model which should to be used for prediction.')

    parser.add_argument('-o', '--output_folder', type=str, default=None, help='Name of the output folder to be created for saving the preprocessed images. The folder is created in ./data/output_images_and_videos. If not set this will be equal to  model_name_epoch from the model which is used.')

    parser.add_argument('-cv', '--conditional_video', type=str, default=None, help='Filename of the video where the images in the input folder are from. Has to end with .mp4 and the video must be in ./data/raw_video. This video is put right next to the fake video in the final result. If not set it will be equal to the name of the test_folder of the  model which is used.')

    parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size for the prediction.')

    parser.add_argument('-d','--dropout_rate', type=float, default=None, help='Dropout rate used in various layers of the GAN. If not set the dropout rate from the trained model is used (as common in GAN prediction).')

    args = parser.parse_args()

    with open(args.trained_model_path.replace('.pth','.json'), 'rt') as f:
        temp_args = argparse.Namespace()
        temp_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=temp_args)

    if args.output_folder is None:
        args.output_folder = args.model_name + '_e' + str(args.current_epoch)
    if args.conditional_video is None:
        args.conditional_video = args.test_folder +'.mp4'

    output_folder_images =os.path.join('./data/output_images_and_videos', args.output_folder, 'images')

    output_folder_video =os.path.join('./data/output_images_and_videos', args.output_folder, 'video')

    if args.input_folder is None:
        if args.test_folder is not None:
            args.__dict__.update({'input_folder': args.test_folder})
        else:
            raise ValueError('Please define an input folder, e.g., -i marcel')
    # Load parameters from the trained model. Command line parameters overwrite the parameters from the file.

    # Make output dirs
    if os.path.exists(output_folder_images):
        shutil.rmtree(output_folder_images)
    if os.path.exists(output_folder_video):
        shutil.rmtree(output_folder_video)
    os.makedirs(os.path.join(output_folder_images, 'fake'))
    os.makedirs(os.path.join(output_folder_images, 'condition'))
    os.makedirs(output_folder_video, exist_ok= True)

    # Parameter Preprocessing -- End

    # Load the Model
    gpu_ids = get_gpu_ids()

    generator = Generator(input_channels=args.number_of_input_channels, output_channels=args.number_of_output_channels, dropout_rate=args.dropout_rate, number_of_blocks=args.number_of_blocks_generator, unet_channels=args.number_of_channels_generator)
    generator.initialize(gpu_ids, args.trained_model_path)

    # You can resize you preprocessed images if necessary a little bit. E.g. if the conditional images have a rather narrow face and the trained images have a rather wide face, you could winden the conditional images a little bit to get prettier results. Little cheating here ;-).

    #transform = transforms.Compose([transforms.CenterCrop((530,2*500)),transforms.Resize((512,2*512)),transforms.ToTensor()])

    transform = transforms.Compose([transforms.CenterCrop((512,2*512)),transforms.Resize((512,2*512)),transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(os.path.join('./data/preprocessed_images/',args.input_folder,'stitched_frames'), transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Predict the fake images and save them
    print('Starting prediction.')
    counter = 0
    batch_member = range(1, args.batch_size+1)
    batch_member = [f'{bm}'.zfill(2) for bm in batch_member]
    for i, (image, _) in  tqdm(enumerate(dataloader, 0), total=math.ceil(len(dataloader.dataset)//args.batch_size)):
        # Load condition
        counter +=1
        condition, _ = load_images_from_stitched(image, args.base_image_size, gpu_ids, testmode=True)

        # Predict
        with torch.no_grad():
            fake = generator(condition)
        counter_padded  = f'{counter}'.zfill(5)

        # Save Image
        for batch_member, one_fake in enumerate(torch.split(fake, 1)):
            save_image(one_fake, os.path.join(output_folder_images, 'fake',  counter_padded + f'{batch_member}'.zfill(2) + '.jpg'))
        for batch_member, one_condition in enumerate(torch.split(condition, 1)):
            save_image(one_condition, os.path.join(output_folder_images, 'condition',  counter_padded + f'{batch_member}'.zfill(2) + '.jpg'))
    print('Finished prediction.')

    # Put everything into a video
    print('Starting video creation. This might take a couple of minutes.')

    # Load images and put them into a list, note that the order is important.
    filenames = [f for f in os.listdir(os.path.join(output_folder_images, 'fake')) if not f.startswith('.')]
    clips = []
    filenames.sort()
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            clips.append(mp.ImageClip(os.path.join(output_folder_images,'fake',filename)))
    nframes=len(clips)

    audio_original = mp.AudioFileClip(os.path.join('./data/raw_video',args.conditional_video))
    target_duration = audio_original.duration/nframes

    # Stretch the images to the desired length and compose them to a video
    clips = [clip.set_duration(target_duration) for clip in clips]
    clip_fake = mp.concatenate_videoclips(clips, method='compose')

    # Put the audio from the original video under the fake video
    new_audioclip = mp.CompositeAudioClip([audio_original])
    clip_fake = clip_fake.set_audio(new_audioclip)

    # Save fake video
    clip_fake.write_videofile(os.path.join(output_folder_video, 'fake.mp4'), fps=1/  target_duration)

    # Combine the two videos
    #fake = mp.VideoFileClip(os.path.join(output_folder_video, 'fake.mp4'))
    original = mp.VideoFileClip(os.path.join('./data/raw_video', args.conditional_video))
    original = original.resize(newsize=(args.base_image_size, args.base_image_size))

    # Comment in those two lines to get rid of any audio
    #fake = fake.set_audio(None)
    #original = fake.set_audio(None)

    combined = mp.clips_array([[original, clip_fake]])
    combined = combined.set_audio(new_audioclip)
    combined.write_videofile(os.path.join(output_folder_video, 'combined.mp4'))

    print(f'Finished video creation. Output in {output_folder_video}.')
