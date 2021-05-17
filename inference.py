from mmdet.apis import init_detector, inference_detector
import mmcv
import matplotlib.pyplot as plt
import cv2
import json
from tqdm.notebook import tqdm
import numpy as np


data = open('../../data/ocr_results/results/ocr_with_gameclockrunning/2018-11-28_Virginia_at_Maryland/2018-11-28_Virginia_at_Maryland_ocr.json')
data = json.load(data)

def convert_to_center(result):
#     print('lol', np.array([int(result[0][2] - result[0][0]), int(result[0][3] - result[0][1])]))
    return np.array([int(result[0][2] - result[0][0]), int(result[0][3] - result[0][1])])

config_file = 'configs/yolo/custom_yolov3_d53_mstrain-608_273e_coco.py'
checkpoint_file = 'work_dirs/yolov3_d53_mstrain-608_273e_coco/epoch_273.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a video and show the results
video = mmcv.VideoReader('../../data/videos/videos/2018-11-28_Virginia_at_Maryland.mp4')

en = 0
proc = 0

current_num_tracks = 0
current_tracks = []  ### List of tracks with history greater than 5
temp_tracks = []  ### List of tracks with history less than 5

temp_track_len = 5
thresh_distance = 0.1
avg_dist = []

pbar = tqdm(total=video.frame_cnt)

for frame in video:

    ### Check if it is game moment
    idx = str(en)
    if not data['results'][idx]['score_bug_present'] or not data['results'][idx]['game_clock_running']:
        en += 1
        continue

    ### Predict the image
    result = inference_detector(model, frame)

    ### Check if we have any detections
    if len(result[0]) == 0:
        en += 1
        current_tracks.append(None)
        temp_tracks = []
        continue

    ### If we have predictions:
    ### First, workout single prediction:
    elif len(result[0]) == 1:

        ### If we don't have approved tracks
        if len(current_tracks) == 0 or current_tracks[-1] is None:
            ### if we don't temporary tracks we update temp tracks or we lost the track, else we check its distance to previous temp track
            if len(temp_tracks) == 0:
                temp_tracks.append(result[0])
            else:
                center = convert_to_center(result[0])
                #                 convert_to_center(temp_tracks[-1])
                #                 print(center, convert_to_center(temp_tracks[-1]))
                #                 print(np.linalg.norm(center - convert_to_center(temp_tracks[-1])))
                if np.linalg.norm(center - convert_to_center(temp_tracks[-1])) < thresh_distance:
                    if len(temp_tracks) < temp_track_len:
                        temp_tracks.append(result[0])
                    else:
                        temp_tracks.append(result[0])
                        current_tracks.extend(temp_tracks)
                        temp_tracks = []

                else:
                    current_tracks.extend([None] * (len(temp_tracks) + 1))
                    temp_tracks = []

        else:
            ### Now, we work with tracks that have history greater than 5
            center = convert_to_center(result[0])

            if np.linalg.norm(center - convert_to_center(current_tracks[-1])) < thresh_distance:
                avg_dist.append(np.linalg.norm(center - convert_to_center(current_tracks[-1])))
                current_tracks.append(result[0])
            else:
                current_tracks.append(None)

    elif len(result[0]) > 1:
        if len(current_tracks) == 0 or current_tracks[-1] is None:
            current_tracks.append(None)
        else:
            for curr_res in result[0]:
                center = convert_to_center(curr_res)
                if np.linalg.norm(center - convert_to_center(current_tracks[-1])) < thresh_distance:
                    current_tracks.append(result[0])
                    continue

            current_tracks.append(None)

    ### If we had too many missed frames, we update the history of the previous frame.

    #     loc = [result[0][0][2] - result[0][0][0], result[0][0][3] - result[0][0][1]]
    #     locations.append(loc)

    #     out_img = model.show_result(frame, result, wait_time=1)

    en += 1
    proc += 1
    if proc > 100: break
    pbar.update(1)
pbar.close()

