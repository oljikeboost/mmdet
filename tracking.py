import numpy as np
import logging

class Tracker():

    def __init__(self):

        self.current_tracks = []  ### List of all tracks
        self.temp_tracks = []  ### List of temporary tracks

        self.temp_track_len = 3  ### How much tracks to collect before merging into main track
        self.thresh_distance = 13  ### Euclidean distance threshold
        self.avg_dist = []

    def convert_to_center(self, result):
        return np.array([int((result[2] + result[0]) / 2), int((result[3] + result[1]) / 2)])

    def update(self, result):

        ### Check if we have any detections
        if len(result[0]) == 0:

            self.current_tracks.append(None)
            self.temp_tracks = []
            self.zero_res = result

        elif len(result[0]) == 1:
            self.single_res = result[0][0]

            logging.debug("SINGLE RES")

            ### If we don't have approved tracks or no good history
            if len(self.current_tracks) == 0 or self.current_tracks[-1] is None:
                logging.debug("NONE TRACK CONTINUED")
                ### if we don't temporary tracks we update temp tracks or we lost the track, else we check its distance to previous temp track
                if len(self.temp_tracks) == 0:
                    self.temp_tracks.append(result[0][0])
                    self.current_tracks.append(None)

                else:
                    curr_center = self.convert_to_center(result[0][0])
                    last_center = self.convert_to_center(self.temp_tracks[-1])
                    self.avg_dist.append(np.linalg.norm(curr_center - last_center))

                    if np.linalg.norm(curr_center - last_center) < self.thresh_distance:
                        logging.debug("DIST CRITERIA SATISFIED")
                        self.temp_tracks.append(result[0][0])

                        if len(self.temp_tracks) < self.temp_track_len:
                            self.current_tracks.append(None)
                        else:
                            logging.debug("CURRENT_TRACK UPDATED-------------------")
                            self.current_tracks.append(None)
                            self.current_tracks[-len(self.temp_tracks):] = self.temp_tracks.copy()
                            self.temp_tracks = []

                    else:
                        self.current_tracks.append(None)
                        self.current_tracks[-(len(self.temp_tracks) + 1):] = [None for _ in
                                                                              range(len(self.temp_tracks) + 1)]
                        self.temp_tracks = []

            else:
                #                 self.temp_tracks = []
                ### Now, we work with tracks that have history greater than 5
                logging.debug("CURRENT TRACK CONTINUED")
                curr_center = self.convert_to_center(result[0][0])
                last_canter = self.convert_to_center(self.current_tracks[-1])

                if np.linalg.norm(curr_center - last_canter) < self.thresh_distance:
                    logging.debug("GOOD TRACK APPENDED")
                    self.avg_dist.append(np.linalg.norm(curr_center - last_canter))
                    self.current_tracks.append(result[0][0])
                else:
                    logging.debug("BAD TRACK APPENDED")
                    self.current_tracks.append(None)

        elif len(result[0]) > 1:
            self.multi_res = result

            logging.debug("MULTI RES")
            #             self.temp_tracks = []

            if len(self.current_tracks) == 0 or self.current_tracks[-1] is None:

                if len(self.temp_tracks) > 0:
                    appended = False

                    for i in range(len(result[0])):

                        curr_res = result[0][i]
                        curr_center = self.convert_to_center(curr_res)
                        last_center = self.convert_to_center(self.temp_tracks[-1])
                        if np.linalg.norm(curr_center - last_center) < self.thresh_distance:

                            self.temp_tracks.append(result[0][0])
                            if len(self.temp_tracks) < self.temp_track_len:
                                self.current_tracks.append(None)
                            else:
                                logging.debug("CURRENT_TRACK UPDATED-------------------")
                                self.current_tracks.append(None)
                                self.current_tracks[-len(self.temp_tracks):] = self.temp_tracks.copy()
                                self.temp_tracks = []

                            appended = True
                            break

                    if not appended:
                        self.current_tracks.append(None)
                        self.current_tracks[-(len(self.temp_tracks) + 1):] = [None for _ in
                                                                              range(len(self.temp_tracks) + 1)]
                        self.temp_tracks = []

                else:
                    self.current_tracks.append(None)

            else:
                #                 self.current_tracks.append(None)
                self.temp_tracks = []
                appended = False
                for i in range(len(result[0])):

                    curr_res = result[0][i]
                    curr_center = self.convert_to_center(curr_res)
                    last_center = self.convert_to_center(self.current_tracks[-1])
                    if np.linalg.norm(curr_center - last_center) < self.thresh_distance:
                        appended = True
                        self.current_tracks.append(curr_res)
                        break

                if not appended:
                    self.current_tracks.append(None)

    def calc_missing_intervals(self, length=2):

        i = 0
        j = 0
        misses = 0
        while i < len(self.current_tracks) and j < len(self.current_tracks):
            if self.current_tracks[i] is not None:
                i += 1
                j = i
            elif self.current_tracks[i] is None and self.current_tracks[j] is None:
                j += 1
            elif self.current_tracks[i] is None and self.current_tracks[j] is not None and 0 < (j - i) <= length and i==0:
                i = j
            elif self.current_tracks[i] is None and self.current_tracks[j] is not None and 0 < (j - i) <= length:
                interp_tracks = self.interpolate_bboxes(self.current_tracks[i - 1:j + 1])
                self.current_tracks[i - 1:j + 1] = interp_tracks
                i = j
                misses += 1
            elif self.current_tracks[i] is None and self.current_tracks[j] is not None and (j - i) > length:
                i = j

        if 0 < (j - i) <= length:
            misses += 1

        return misses

    def interpolate_bboxes(self, tracks):

        first = tracks[0]
        last = tracks[-1]

        first_center = [(first[2] + first[0]) / 2, (first[3] + first[1]) / 2]
        last_center = [(last[2] + last[0]) / 2, (last[3] + last[1]) / 2]

        half_width = ((abs(first[2] - first[0]) + abs(last[2] - last[0])) / 2) / 2
        half_height = ((abs(first[3] - first[1]) + abs(last[3] - last[1])) / 2) / 2

        i = 1

        while i < len(tracks) - 1:
            new_x = first_center[0] + ((i + 1) / len(tracks)) * (last_center[0] - first_center[0])
            new_y = first_center[1] + ((i + 1) / len(tracks)) * (last_center[1] - first_center[1])

            new_bbox = np.array(
                [new_x - half_width, new_y - half_height, new_x + half_width, new_y + half_height, first[-1]])

            tracks[i] = new_bbox

            i += 1

        return tracks

