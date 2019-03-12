import time
import functools
import logging
import argparse

import cv2
import os
import skimage.color
import skimage.io
import skvideo.io
import dlib
import numpy as np
from collections import OrderedDict


def timer(func):
    @functools.wraps(func)
    def wraper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('{0} took: {1} seconds'.format(func.__name__, (end - start)))
        return result

    return wraper


def separete_video(video_path, img_dir):
    video = skvideo.io.vread(video_path)
    for i, frame in enumerate(video):
        skimage.io.imsave(os.path.join(img_dir, '{:0>4d}.jpg'.format(i)),
                          frame)


class FaceAligner:
    def __init__(self, output_shape=256,
                 predictor_path='./shape_predictor_68_face_landmarks.dat'):
        if not os.path.isfile(predictor_path):
            predictor_path = os.path.join(os.path.dirname(__file__), predictor_path)
        self.predictor = dlib.shape_predictor(predictor_path)
        self.detector = dlib.get_frontal_face_detector()

    def _detect_biggest(self, img, preprocess_func=lambda image: image):
        def rect_area(rect):
            width = rect.right() - rect.left()
            height = rect.height() - rect.bottom()
            return abs(width * height)

        face_rects = self.detector(preprocess_func(img))
        if not face_rects:
            return None
        face_rects_area = [rect_area(rect) for rect in face_rects]
        return face_rects[np.argmax(face_rects_area)]

    def face_landmark(self, img):
        face_rect = self._detect_biggest(img)
        if face_rect is not None:
            landmark = self.predictor(img, face_rect)
            landmark = np.array([[p.x, p.y] for p in landmark.parts()])
        else:
            landmark = np.array([])
        return landmark

    def translate_to_landmarks(self, img_dir, keypoints_dir):
        img_base_dir = os.path.basename(img_dir)
        save_dir = os.path.join(keypoints_dir, img_base_dir)
        os.makedirs(save_dir)

        filenames = os.listdir(img_dir)
        try:
            for filename in sorted(filenames):
                if not filename.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                print(os.path.join(img_dir, filename))
                img = skimage.io.imread(os.path.join(img_dir, filename))
                landmark = self.face_landmark(img)
                print(landmark.shape)
                np.savetxt(os.path.join(save_dir, filename + '.txt'), landmark, delimiter=',')
        except:
            return False
        return True

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_dir', type=str, help='where the face img stored')
        parser.add_argument('--keypoints_dir', type=str, help='where to save teh face landmark detected')
        args = parser.parse_args()
        # separete_video(args.video_path, args.img_dir)

        aligner = FaceAligner()
        aligner.translate_to_landmarks(args.img_dir, args.keypoints_dir)

    if __name__ == '__main__':
        main()
        # aligner = FaceAligner()
