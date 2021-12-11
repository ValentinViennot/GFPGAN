#!/usr/bin/env python3
import cv2
import numpy as np
import os
import shutil
import time
import hashlib

from threading import Thread

from flask import Flask, Response, abort, request, send_from_directory
from flask_cors import CORS

from basicsr.utils import imwrite
from gfpgan import GFPGANer

api = Flask(__name__)
save_root = 'results'
CORS(api)

def original_path(name, root=save_root):
    return os.path.join(root, f'{name}', 'original.png')

def restored_path(name, root=save_root):
    return os.path.join(root, f'{name}', 'restored.png')

def faces_dir(name, root=save_root):
    return os.path.join(root, f'{name}', 'faces')

def faces_path(name, idx):
    return os.path.join(faces_dir(name), f'{idx:02d}.png')

def md5(filebytes):
    hash_md5 = hashlib.md5()
    hash_md5.update(filebytes)
    return hash_md5.hexdigest()

def clean_old_results(name):
    try:
        os.remove(restored_path(name))
    except OSError as e:
        print('nothing to clean')
    try:
        shutil.rmtree(faces_dir(name))
    except OSError as e:
        print('nothing to clean')

def import_file(file, cv2_img_flag=cv2.IMREAD_ANYCOLOR):
    filebytes = file.read()
    img_array = np.asarray(bytearray(filebytes), dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2_img_flag)
    name = md5(filebytes)
    clean_old_results(name)
    imwrite(img_cv, original_path(name))
    return (img_cv, name)

def load_image(name, cv2_img_flag=cv2.IMREAD_COLOR):
    clean_old_results(name)
    img_path = original_path(name)
    if not os.path.isfile(img_path):
        abort(404, 'File not found')
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return input_img

def convert_to_bytes(open_cv_img):
    _, buffer = cv2.imencode('.png', open_cv_img)
    return buffer.tobytes()

def restore(input_img, image_name, hide_faces):
    restorer = GFPGANer(
        model_path='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)
    _, restored_faces, restored_img = restorer.enhance(
            input_img, has_aligned=False, only_center_face=False, paste_back=True, hide_faces=hide_faces)
    # save faces
    for idx, restored_face in enumerate(restored_faces):
        imwrite(restored_face, faces_path(image_name, idx))
    # save restored image
    imwrite(restored_img, restored_path(image_name))

def get_image_files(name):
    if not os.path.isfile(original_path(name)):
        abort(404, 'Hash not found in memory.')
    original = '/original.png'
    restored = '/restored.png' if os.path.isfile(restored_path(name)) else None
    faces = os.listdir(faces_dir(name)) if os.path.isdir(faces_dir(name)) else []
    return {
        'hash': name,
        'original': original,
        'restored': restored,
        'faces': [ f'/faces/{face}' for face in faces ],
    }

@api.route('/restore', methods=['POST'])
def post_image():
    form_data = request.form.to_dict()
    image_name = form_data['hash'] if 'hash' in form_data else None
    hide_faces = [ int(i) for i in form_data['hide_faces'].split(',') ] if 'hide_faces' in form_data else []
    if 'image' in request.files:
        input_img, image_name = import_file(request.files['image'])
    elif image_name is not None:
        input_img = load_image(image_name)
    else:
        abort(400, 'Either upload an image or pass a hash.')
    # start the restoration process asynchronously
    thread = Thread(target=restore, kwargs={
        'input_img': input_img,
        'image_name': image_name,
        'hide_faces': hide_faces
    })
    thread.start()
    # return data about the image being processed
    return get_image_files(image_name)

@api.route('/restore/<name>')
def get_image(name):
    return get_image_files(name)

@api.route('/img/<path:path>')
def send_image(path):
    return send_from_directory('results', path)

if __name__ == '__main__':
    api.run(host="0.0.0.0")

# docker build . -t gfpgan
# docker run -p 5000:5000 gfpgan
# while developing, run with:
# docker run -p 5000:5000 -v $(pwd)/server.py:/app/server.py -e FLASK_APP=server.py -e FLASK_ENV=development gfpgan