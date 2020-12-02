import requests
import shutil
import os


#taken from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python/blob/master/Download-Large-File-from-Google-Drive.ipynb
# and https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == '__main__':

    dl_path_facial_landmark_model = '12cTovCMw48PV8ncNPOSQKPu62l45eMIy'
    dl_path_videos = '12cTovCMw48PV8ncNPOSQKPu62l45eMIy'

    facial_landmark_tmp = './tmp/facial_landmark_model.tar.gz'
    facial_landmark_dest = './data/facial_landmark_model'
    videos_tmp = './tmp/videos.tar.gz'
    videos_dest = './data/raw_video'

    temp_folder_path = './tmp'

    os.makedirs(temp_folder_path, exist_ok=True)
    print('./tmp folder created.')

    download_file_from_google_drive(dl_path_facial_landmark_model, facial_landmark_tmp)
    download_file_from_google_drive(dl_path_videos, videos_tmp)

    print('Files downloaded.')

    shutil.unpack_archive(facial_landmark_tmp, facial_landmark_dest)
    print(f'File {facial_landmark_tmp} extracted to {facial_landmark_dest}.')

    shutil.unpack_archive(videos_tmp, videos_dest)
    print(f'File {videos_tmp} extracted to {videos_dest}.')

    shutil.rmtree(temp_folder_path)
    print('./tmp folder deleted.')
    print('Data downloads successfully finished.')

