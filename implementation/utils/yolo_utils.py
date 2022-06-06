import cv2
import numpy as np

class YoloBb:
    def __init__(self, frame, cell_idx, cell_idy, yolo_array, box_id, anchor_values, class_labels):
        assert len(yolo_array) == 85, 'Not a valid box. Boxes contain 85 entries.'
        # extend of a cell in pixels
        cell_dim = [frame.shape[0] / 13.0, frame.shape[1] / 13.0]
        # cell id along the x and y-axis in the range of [0:13]
        self.cell_idx = cell_idx 
        self.cell_idy = cell_idy 
        # relative center position within a cell
        c_x_rel = self.sigmoid(yolo_array[0])
        c_y_rel = self.sigmoid(yolo_array[1])
        # absolute center position of yolo bounding box
        self.c_x = cell_dim[0] * self.cell_idx + c_x_rel * cell_dim[0]
        self.c_y = cell_dim[1] * self.cell_idy + c_y_rel * cell_dim[1]
        # height and width of a bounding box, three possible dimensions per cell
        self.h = np.exp(yolo_array[2]) * anchor_values[box_id*2] * 2
        self.w = np.exp(yolo_array[3]) * anchor_values[box_id*2+1] * 2
        # confidence of a bounding box, needed to determine best candidate per cell from three anchor boxes per cell
        self.box_conf = self.sigmoid(yolo_array[4])
        # information about the class detected in the bounding box 
        self.class_id = np.argmax(yolo_array[5:85])
        self.class_conf = self.box_conf * self.sigmoid(max(yolo_array[5:85]))
        self.class_label = class_labels[self.class_id]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __str__(self):
        return f'label: {self.class_label} ({round(self.class_conf * 100.0, 2)}%)\ncell (x, y): ({self.cell_idx}, {self.cell_idy})\nabs_center (x, y): ({self.c_x, self.c_y})\nbb (h, w): ({self.h}, {self.w})'

class YoloBbExtractor():
    def __init__(self, class_labels, anchor_values, nms_class_threshold, nms_iou_threshold):
        self.class_labels = class_labels
        self.anchor_values = anchor_values
        self.nms_class_threshold = nms_class_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def get_yolo_bbs(self, frame, results, class_id_to_include, anchor_values):
        '''
        Returns all YOLO boundary boxes of a frame matching a given class id.
        '''
        # for Yolo v4 - example model description including expected output format
        # https://docs.openvino.ai/latest/omz_models_model_yolo_v4_tiny_tf.html

        # 
        # How to interpret the output of the compiled Openvino Yolo v4 tinf tf model:
        #
        # https://arxiv.org/abs/2004.10934 (YOLOv4)
        #
        # # https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/yolo-v4-tiny-tf/README.md
        #
        # result as [B, Cx, Cy, N * 85] with N := 3
        # [1, 13, 13, 255] or [1, 13, 13, 3 * 85]
        #
        # B - batch size
        # N - number of detection boxes for cell
        # Cx, Cy - cell index
        # 
        # each of the three detection boxes in one of the 13x13 grid slots has the form
        # 
        # [x, y, h, w, box_score, class_no_1, …, class_no_80]  -> len := 85
        y_objects = []
        # predetermined by YOLO
        x_cells_per_frame = 13
        y_cells_per_frame = 13
        n_bb_per_cell = 3
        array_entries_per_bb = 85
        
        for cell_idx in range(0, x_cells_per_frame):
            for cell_idy in range(0, y_cells_per_frame):
                for box_id in range(0, n_bb_per_cell):
                    box_start = box_id * array_entries_per_bb
                    box_stop = (box_id + 1) * array_entries_per_bb
                    box = results[0, cell_idx, cell_idy, box_start:box_stop]
                    y = YoloBb(frame, cell_idx, cell_idy, box, box_id, anchor_values, self.class_labels)
                    # filter out classes that are of no interest
                    if y.class_id == class_id_to_include:
                        y_objects.append(y)
        return y_objects

    def get_nms_bbs(self, yolo_bbs, class_threshold, nms_threshold=0.5):
        class_ids, scores, nms_boxes = [], [], []
        # convert to nms bounding boxes
        for yolo_bb in yolo_bbs:
            class_ids.append(yolo_bb.class_id)
            scores.append(yolo_bb.box_conf)
            nms_box = [yolo_bb.c_y, yolo_bb.c_x, yolo_bb.h, yolo_bb.w]
            nms_boxes.append(nms_box)
        # filter out irrelevant bounding boxes
        nms_idxs = cv2.dnn.NMSBoxes(nms_boxes, scores, class_threshold, nms_threshold)
        # return empty if no more bounding boxes
        if len(nms_idxs) == 0:
            return []
        # return nms bounding boxes
        return [(class_ids[idx], scores[idx], nms_boxes[idx]) for idx in nms_idxs.flatten()]

    def extract(self, frame, results, class_id_to_include):
        yolo_bbs = self.get_yolo_bbs(frame, results, class_id_to_include, self.anchor_values)
        bbs_drawing = self.get_nms_bbs(yolo_bbs, self.nms_class_threshold, self.nms_iou_threshold)
        bbs_tracking = [bb_tracking[2] for bb_tracking in bbs_drawing]
        return (bbs_tracking, bbs_drawing)