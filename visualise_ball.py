for frame in video:
    pbar.update(1)

    if all_tracks[proc] is not None:

        result = all_tracks[proc][:4]
        center = (int((result[0] + result[2]) / 2), int((result[1] + result[3]) / 2))
        radius = int((result[2] - result[0]) / 2)
        cv2.circle(frame, center, radius, (0, 255, 0), thickness=1, lineType=8, shift=0)

        if all_tracks[proc-1] is not None:

            curr = all_tracks[proc].astype(np.int32)
            prev = all_tracks[proc-1].astype(np.int32)
            if abs(curr[0] - prev[0]) < 10:



                dist_to_center = int(math.sqrt((curr[2] - curr[0])**2 + (curr[3] - curr[1])**2) / 2)
                dist_to_circle = int((dist_to_center - radius) / math.sqrt(2))

                new_curr = [curr[0] + dist_to_circle, curr[1] + dist_to_circle,
                           curr[2] - dist_to_circle, curr[3] - dist_to_circle]

                prev_radius = int((prev[2] - prev[0]) / 2)
                dist_to_center = int(math.sqrt((prev[2] - prev[0])**2 + (prev[3] - prev[1])**2) / 2)
                dist_to_circle = int((dist_to_center - prev_radius) / math.sqrt(2))

                new_prev = [prev[0] + dist_to_circle, prev[1] + dist_to_circle,
                           prev[2] - dist_to_circle, prev[3] - dist_to_circle]


                #upper left
                cv2.line(frame, (new_curr[0], new_curr[1]), (new_prev[0], new_prev[1]), (0, 255, 0), thickness=l_thickness)
                #upper right 
                cv2.line(frame, (new_curr[2], new_curr[1]), (new_prev[2], new_prev[1]), (0, 255, 0), thickness=l_thickness)
                #lower left
                cv2.line(frame, (new_curr[0], new_curr[3]), (new_prev[0], new_prev[3]), (0, 255, 0), thickness=l_thickness)
                #lower right
                cv2.line(frame, (new_curr[2], new_curr[3]), (new_prev[2], new_prev[3]), (0, 255, 0), thickness=l_thickness)

