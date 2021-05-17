from mmdet.apis import init_detector, inference_detector
import mmcv
import matplotlib.pyplot as plt
import cv2
import os
import json
from tqdm.notebook import tqdm
import numpy as np
import logging
from tracking import Tracker

### Path to OCR of the video
data = open('../../data/ocr_results/results/ocr_with_gameclockrunning/2018-11-28_Virginia_at_Maryland/2018-11-28_Virginia_at_Maryland_ocr.json')
data = json.load(data)

### Config and model weights path
config_file = 'configs/yolo/custom_yolov3_d53_mstrain-608_273e_coco.py'
checkpoint_file = 'work_dirs/yolov3_d53_mstrain-608_273e_coco/epoch_273.pth'

### Path to the video
video_path = '../../data/videos/videos/2018-11-28_Virginia_at_Maryland.mp4'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

logging.basicConfig(level=logging.DEBUG)
### Comment if you want to DEBUG
logging.disable(logging.DEBUG)

### Load the video using MMCV
video = mmcv.VideoReader(video_path)

### Temp variables
en = 0
proc = 0
pbar = tqdm(total=video.frame_cnt)
has_pred = 0

### Create the tracker
tracker = Tracker()

### Start prediction
for frame in video:

    ### Check if it is game moment using OCR
    idx = str(en)
    if not data['results'][idx]['score_bug_present'] or not data['results'][idx]['game_clock_running']:
        en += 1
        continue

    ### Predict the image and update stats
    result = inference_detector(model, frame)
    if len(result[0]) != 0: has_pred += 1

    ### Update tracker results
    tracker.update(result)

    ### Update stats
    en += 1
    proc += 1
    if proc > 600: break
    pbar.update(1)

pbar.close()


### Here, I check the number of collected detection and processed frames
print(len(tracker.current_tracks), proc)

### Here I run the function which search for missing detections intervals in tracker.current_tracks
### and interpolates its values. The below threshold is pointing how many missed values to interpolate.
how_much_to_interpoate = 6
tracker.calc_missing_intervals(how_much_to_interpoate)



# test a video and show the results
video = mmcv.VideoReader('../../data/videos/videos/2018-11-28_Virginia_at_Maryland.mp4')
out = cv2.VideoWriter('output_tracker.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (1280, 720))

en = 0
proc = 0
not_none = 0
for frame in video:
    idx = str(en)
    if not data['results'][idx]['score_bug_present'] or not data['results'][idx]['game_clock_running']:
        en += 1
        continue

    if tracker.current_tracks[proc] is not None:
        not_none += 1
        result = np.expand_dims(tracker.current_tracks[proc][:4], axis=0)
        frame = mmcv.imshow_bboxes(frame, result, show=False, thickness=1, colors=['green'])

    out.write(frame)

    en += 1
    proc += 1
    if proc > 500: break

out.release()
