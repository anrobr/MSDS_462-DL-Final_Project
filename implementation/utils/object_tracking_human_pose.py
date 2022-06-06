import numpy as np
import collections
import cv2
import time
import matplotlib.pyplot as plt

from IPython import display
from decoder import OpenPoseDecoder
from object_tracking import ObjectTracker
from visualization import VideoPlayer

# 2d pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = np.lib.stride_tricks.as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)

# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)

# get poses from results
def process_results(model, img, pafs, heatmaps):
    # this processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)
    # decode poses
    decoder = OpenPoseDecoder()
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
    # multiply coordinates by scaling factor
    poses[:, :, :2] *= output_scale
    return poses, scores

def draw_poses(object_tracker, frame, frame_nr, poses, point_score_threshold, skeleton, colors, body_parts_to_colors, body_part_to_track, bb_color, alpha, show_frame=False):
    if poses.size == 0:
        return ([object_tracker.counter, object_tracker.objects], frame, 0)

    img_limbs = np.copy(frame)
    scores, centroids = [], []
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        # Draw joints
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                if colors[i] == body_parts_to_colors[body_part_to_track]:
                    scores.append(np.round(v, 2))
                    centroids.append(np.array([p[0], p[1], 0, 0]))
                    cv2.circle(frame, tuple(p), 1, colors[i], 10)
        # Draw limbs
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
    cv2.addWeighted(frame, alpha, img_limbs, 1-alpha, 0, dst=frame)

    tracking_info = object_tracker.update(centroids, frame_nr)

    for id, pos in tracking_info[1].items():
        cv2.putText(img=frame, text=f'(x,y): {(pos[0], pos[1])}', org=(pos[0], pos[1] - 80), 
            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=frame.shape[1] / 1500, color=bb_color,
            thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img=frame, text=f'Id:    {id}', org=(pos[0], pos[1] - 100), 
            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=frame.shape[1] / 1500, color=bb_color,
            thickness=1, lineType=cv2.LINE_AA)

    if show_frame:
        plt.figure(figsize = (12,12))
        plt.imshow(frame)

    return tracking_info, frame, len(centroids)

def infer_pose(model, frame, width, height):
    # resize image and change dims to fit neural network input
    # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
    input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    # create batch of images (size = 1)
    input_img = input_img.transpose((2,0,1))[np.newaxis, ...]   
    start_time = time.time()
    results = model([input_img])
    stop_time = time.time()
    return (stop_time - start_time), results

# main processing function to run pose estimation
def run_pose_estimation(
    model, 
    source,
    frame_roi_xmin,
    frame_roi_xmax, 
    img_width, 
    img_height,
    point_score_threshold,
    skeleton, 
    colors,
    body_parts_to_colors, 
    body_part_to_track, 
    bb_color, 
    bb_alpha,
    allowed_lost_tracking_frames,
    flip=False, 
    use_popup=False, 
    skip_first_frames=0):

    pafs_output_key = model.output("Mconv7_stage2_L1")
    heatmaps_output_key = model.output("Mconv7_stage2_L2")

    player = None
    try:
        frame_nr = 0
        object_tracker = ObjectTracker(allowed_lost_tracking_frames)
        detected_objects_counter = {}
        # create video player to play with target fps
        player = VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # start capturing
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()

        while True:
            # grab the frame
            frame = player.next()
            frame_nr += 1

            if frame is None:
                print("\nSource ended")
                break

            frame = frame[:, frame_roi_xmin:frame_roi_xmax]
            elapsed_time, results = infer_pose(model, frame, img_width, img_height)
            pafs = results[pafs_output_key]
            heatmaps = results[heatmaps_output_key]

            # get poses from network results
            poses, _ = process_results(model, frame, pafs, heatmaps)
            # draw poses on a frame
            tracking_info, _, detected_objects_count = draw_poses(object_tracker, frame, frame_nr, poses, point_score_threshold, skeleton, colors, body_parts_to_colors, body_part_to_track, bb_color, bb_alpha, False)
            
            if detected_objects_count in detected_objects_counter:
                detected_objects_counter[detected_objects_count] += 1
            else:
                detected_objects_counter[detected_objects_count] = 1

            processing_times.append(elapsed_time)
            if len(processing_times) > 90:
                processing_times.popleft()

            _, f_width = frame.shape[:2]

            # frame counter
            cv2.putText(img=frame, text=f'Frame: {frame_nr}', org=(20, 20),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1500,
                        color=bb_color, thickness=1, lineType=cv2.LINE_AA)

            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1500, bb_color, 1, cv2.LINE_AA)

            # person counter
            cv2.putText(img=frame, text=f'Person(s) in maze: {tracking_info[0]}', org=(20, 80),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1500,
                        color=bb_color, thickness=1, lineType=cv2.LINE_AA)

            # use this workaround if there is flickering
            if use_popup:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # encode numpy array to jpg
                _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                # create IPython image
                i = display.Image(data=encoded_img)
                # display the image in this notebook
                display.clear_output(wait=True)
                display.display(i)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        print(f'{player.get_frame_count()} frames total.')
        for k, v in detected_objects_counter.items():
            print(f'{v} frames with {k} object(s).')

        if player is not None:
            # stop capturing
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()