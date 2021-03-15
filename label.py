
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import io
import argparse
import json
import base64

import cv2
import torch
import numpy as np
from glob import glob
import PIL
from PIL import Image

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--label', required=True, type=str)
parser.add_argument('--decimate', required=True, type=int)
parser.add_argument('--output', required=True, type=str)
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

# taken from labelme source


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get("Orientation", None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        assert False, "Failed to load image " + filename
        return

    # apply orientation to image according to exif
    image_pil = apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        image_pil.save(f, format="PNG")
        f.seek(0)
        return f.read()


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    # setup annotations
    anns = []

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for idx, frame in enumerate(get_frames(args.video_name)):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            frame_pre = frame.copy()
            outputs = tracker.track(frame)
            ann = {
                'version': '4.5.7',
                'flags': {},
                'imageData': "000000000",
                'imageHeight': frame_pre.shape[-3],
                'imageWidth': frame_pre.shape[-2]
            }
            ann['shapes'] = [{
                'label': args.label,
                'group_id': None,
                'flags': {}
            }]
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)

                # annotations
                ann['shapes'][0]['shape_type'] = 'polygon'
                maskbg = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # 10 is used arbitrarly as a value greater than 1 since it appears some masks return 1 rather than 0 for some reason
                ret, tr = cv2.threshold(maskbg, 10, 255, 0)
                contours, _ = cv2.findContours(
                    tr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = [d[0] for d in contours[0].tolist()]
                if args.decimate > 1:
                    contours = contours[::args.decimate]
                ann['shapes'][0]['points'] = contours
            else:
                raise Exception('bbox annotations have not been implemented')
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            # save annotations
            output_name = video_name + "_" + str(idx).zfill(9)
            output_path = os.path.join(args.output, output_name)
            ann['imagePath'] = output_name + ".png"

            if not os.path.isdir(args.output):
                os.makedirs(args.output)

            cv2.imwrite(output_path + ".png", frame_pre)
            ann['imageData'] = base64.b64encode(
                load_image_file(output_path + ".png")).decode("utf-8")

            with open(output_path + ".json", "w") as f:
                f.write(json.dumps(ann))

            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()
