import os
import shutil
import numpy as np
import dlib
import cv2
from imutils import video
import argparse
from pdb import set_trace
import time
from tqdm import tqdm

def reshape_landmark_group(landmarks):
    '''
    Reshapes the landmarks belonging to one group (e.g. left eye) such that a polyline
    can be drawn to link them.
    Parameters:
        landmarks: landmarks belonging to one group
    '''
    return np.array(landmarks,np.int32).reshape(-1,1,2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_file', type=str, required=True, help='Name of the input video file. Has to end with *.mp4. File has to be in ./data/raw_video.')

    parser.add_argument('-df', '--desired_frames', required=True, type=int, help='Desired number of output frames. The algorithm targets this number by slicing the input video in an equally distanced manner. Rule of thumb is ~10 frames per second for decent quality.')

    parser.add_argument('-o', '--output_folder', type=str, help='Name of the output folder to be created for saving the preprocessed images. The folder is created in ./data/preprocessed_images.')

    parser.add_argument('-ts', '--target_size', type=int, default=512, help='Desired height and width of the output image.')

    args = parser.parse_args()

    if args.output_folder is None:
        args.__dict__.update({'output_folder':args.input_file.replace('.mp4','')})

    #Initialize parameters
    file_name = os.path.join('./data/raw_video',args.input_file)
    output_folder = os.path.join('./data/preprocessed_images',args.output_folder)
    desired_number_of_frames = args.desired_frames

    output_extension = 'jpg'
    path_to_landmark_model = "./data/facial_landmark_model/shape_predictor_68_face_landmarks.dat"
    target_size = args.target_size

    # Get face detector and landmark predictor
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(path_to_landmark_model)

    # Make the output dirs (overwrite existing one)
    os.makedirs(os.path.join(output_folder,'stitched_frames','all'), exist_ok= True)

    # Get the video
    vidcap = cv2.VideoCapture(file_name)

    # Generate the array for the target frames
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames =np.arange(frame_count, step = frame_count// desired_number_of_frames)

    print('Starting facial landmark detection.')
    for current_frame_number in tqdm(target_frames):
        # Capture the desired frame
        vidcap.set(1,current_frame_number-1)
        success, frame = vidcap.read(1)

        # Convert in case the video is colored
        one_channel_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the face boxes in the frame
        face_boxes = face_detector(one_channel_frame, 1)

        # Proceed to landmark detection
        if len(face_boxes)==1:
            for face in face_boxes:
                landmarks_raw= landmark_predictor(one_channel_frame, face).parts()
                landmarks = [[l.x, l.y] for l in landmarks_raw]

                # Black empty canvas
                black_canv = np.zeros(frame.shape, np.uint8)

                # Get the different parts of the landmarks
                jaw = reshape_landmark_group(landmarks[0:17])
                right_eyebrow = reshape_landmark_group(landmarks[17:22])
                left_eyebrow = reshape_landmark_group(landmarks[22:27])
                upper_nose = reshape_landmark_group(landmarks[27:31])
                lower_nose = reshape_landmark_group(landmarks[30:35])
                right_eye = reshape_landmark_group(landmarks[36:42])
                left_eye = reshape_landmark_group(landmarks[42:48])
                outer_lip = reshape_landmark_group(landmarks[48:60])
                inner_lip = reshape_landmark_group(landmarks[60:68])

                # Draw the lines onto the black canvas
                line_color = (255, 255, 255) #white
                line_thickness = 2

                cv2.polylines(black_canv, [jaw], False, line_color, line_thickness)
                cv2.polylines(black_canv, [right_eyebrow], False, line_color, line_thickness)
                cv2.polylines(black_canv, [left_eyebrow], False, line_color, line_thickness)
                cv2.polylines(black_canv, [upper_nose], False, line_color, line_thickness)
                cv2.polylines(black_canv, [lower_nose], True, line_color, line_thickness)
                cv2.polylines(black_canv, [right_eye], True, line_color, line_thickness)
                cv2.polylines(black_canv, [left_eye], True, line_color, line_thickness)
                cv2.polylines(black_canv, [outer_lip], True, line_color, line_thickness)
                cv2.polylines(black_canv, [inner_lip], True, line_color, line_thickness)

                frame = cv2.resize(frame, (target_size,target_size))
                black_canv = cv2.resize(black_canv, (target_size,target_size))

                # Save the file
                current_frame_number_padded  = f'{current_frame_number}'.zfill(5)

                # Stitch the two images together (so later a single PyTorch dataloader can be used)
                black_canv_frame = np.concatenate([black_canv, frame], 1)
                cv2.imwrite(os.path.join(output_folder,f'stitched_frames/all/{current_frame_number_padded}.{output_extension}'),black_canv_frame)
        elif len(face_boxes)>1:
            print('Multiple faces were detected.')
        else:
            print('No face was detected')
    print('Finished facial landmark detection. Images saved under' + os.path.join(output_folder,'stitched_frames/all/') + '.')
