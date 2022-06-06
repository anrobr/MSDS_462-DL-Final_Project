import cv2
import time
import numpy as np
import collections
import  matplotlib.pyplot as plt

from yolo_utils import YoloBbExtractor
from sddlite_utils import SddLiteMobileNetV2Extractor
from object_tracking import ObjectTracker
from visualization import VideoPlayer

def infer_bbox(compiled_model, frame, frame_height, frame_width, output_layer):
    input_img = cv2.resize(src=frame, dsize=(frame_height, frame_width), interpolation=cv2.INTER_AREA)
    # create batch of images (size = 1)
    input_img = input_img[np.newaxis, ...]
    return compiled_model([input_img])[output_layer]

def draw_bboxes(frame, bbs, tracking_info, class_labels, bb_color, bb_alpha, show_frame):  
    for class_id, score, box in bbs:
        box = list(map(int, box[:4]))
        x_center = int(box[0])
        y_center = int(box[1])
        x_start = int(x_center - box[2] / 2.0)
        y_start = int(y_center - box[3] / 2.0)
        # detection info
        output = frame.copy()
        frame = cv2.rectangle(frame, (x_start, y_start), (x_start + box[2], y_start + box[3]), color=bb_color, thickness=-1)
        cv2.addWeighted(output, bb_alpha, frame, 1 - bb_alpha, 0, frame)
        cv2.putText(img=frame, text=f'Cls:   {class_labels[class_id]} {score:.2f}', org=(x_start, y_start - 60),
                   fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=frame.shape[1] / 1500, color=bb_color,
                   thickness=1, lineType=cv2.LINE_AA)
        # tracking info
        bb_center = (int(box[0]), int(box[1]))
        frame = cv2.circle(frame, bb_center, radius=0, color=bb_color, thickness=5)
        
        for key, value in tracking_info[1].items():
            if bb_center[0] == value[0] and bb_center[1] == value[1]:
                cv2.putText(img=frame, text=f'(x,y): {(x_center, y_center)}', org=(x_start, y_start - 40), 
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=frame.shape[1] / 1500, color=bb_color,
                    thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(img=frame, text=f'Id:    {key}', org=(x_start, y_start - 20), 
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=frame.shape[1] / 1500, color=bb_color,
                    thickness=1, lineType=cv2.LINE_AA)
        if show_frame:
            plt.imshow(frame)
    return frame

def run_object_tracking(
    source,
    frame_roi_xmin,
    frame_roi_xmax,
    model,
    model_name,
    model_input_width,
    model_input_height,
    class_labels,
    class_id_to_include,
    anchor_values,
    nms_class_threshold,
    nms_iou_threshold,
    bb_color, 
    bb_alpha,
    flip,
    skip_first_frames,
    tracker_allowed_frames_lost):
    
    player = None
    try:
        frame_nr = 0
        object_tracker = ObjectTracker(tracker_allowed_frames_lost)
        detected_objects_counter = {}
        # create video player to play with target fps
        player = VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # start capturing
        player.start()
        title = "Press ESC to Exit"
        cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        if model_name == 'yolo-v4-tiny-tf':
            bb_extractor = YoloBbExtractor(class_labels, anchor_values, nms_class_threshold, nms_iou_threshold)
        elif model_name == 'ssdlite_mobilenet_v2':
            bb_extractor = SddLiteMobileNetV2Extractor(nms_class_threshold, nms_iou_threshold)

        processing_times = collections.deque()
        while True:
            # grab the frame
            frame = player.next()
            frame_nr += 1
            if frame is None:
                print("\nSource ended")
                break

            # resize image and change dims to fit neural network input
            frame = frame[:, frame_roi_xmin:frame_roi_xmax]
            input_img = cv2.resize(src=frame, dsize=(model_input_width, model_input_height), interpolation=cv2.INTER_AREA)
            # create batch of images (size = 1)
            if 'pose' not in model_name:
                input_img = input_img[np.newaxis, ...]
            else:
                input_img = input_img.transpose(2,0,1)[np.newaxis, ...]

            start_time = time.time()
            # inference
            if 'pose' not in model_name:
                results = model([input_img])[model.output(0)]
            else:
                results = model([input_img])
            stop_time = time.time()
            
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 90:
                processing_times.popleft()

            # post processing
            bbs_tracking, bbs_drawing = bb_extractor.extract(frame, results, class_id_to_include)       

            detected_objects_count = len(bbs_drawing)
            if detected_objects_count in detected_objects_counter:
                detected_objects_counter[detected_objects_count] += 1
            else:
                detected_objects_counter[detected_objects_count] = 1

            # tracking
            tracking_info = object_tracker.update(bbs_tracking, frame_nr)

            # visualization
            frame = draw_bboxes(frame, bbs_drawing, tracking_info, class_labels, bb_color, bb_alpha, False)
            _ , f_width = frame.shape[:2]

            # frame counter
            cv2.putText(img=frame, text=f'Frame: {frame_nr}', org=(20, 20),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1500,
                        color=bb_color, thickness=1, lineType=cv2.LINE_AA)

            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(img=frame, text=f'Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)', org=(20, 40),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1500,
                        color=bb_color, thickness=1, lineType=cv2.LINE_AA)

            # person counter
            cv2.putText(img=frame, text=f'Person(s) in maze: {tracking_info[0]}', org=(20, 80),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1500,
                        color=bb_color, thickness=1, lineType=cv2.LINE_AA)

            cv2.imshow(winname=title, mat=frame)
            key = cv2.waitKey(1)
            if key == 27: 
                # esc
                break
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
            cv2.destroyAllWindows()