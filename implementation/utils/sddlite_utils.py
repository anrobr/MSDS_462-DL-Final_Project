import cv2

class SddLiteMobileNetV2Extractor():
    def __init__(self, nms_class_threshold, nms_iou_threshold):
        self.nms_class_threshold = nms_class_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def get_nms_bbs(self, frame, results, class_id_to_include):
        h, w = frame.shape[:2]
        results = results.squeeze()
        boxes, labels, scores = [], [], []
        for _, label, score, xmin, ymin, xmax, ymax in results:
            if label == class_id_to_include:
                box = ((xmin * w + xmax * w) / 2.0, (ymax * h + ymin * h) / 2.0, xmax * w - xmin * w, ymax * h - ymin * h)
                boxes.append(box)
                labels.append(int(label))
                scores.append(float(score))
        indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=self.nms_class_threshold, nms_threshold=self.nms_iou_threshold)
        if len(indices) == 0:
            return []
        return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

    def extract(self, frame, results, class_id_to_include):
        bbs_drawing = self.get_nms_bbs(frame, results, class_id_to_include)
        bbs_tracking = [bb_tracking[2] for bb_tracking in bbs_drawing]
        return (bbs_tracking, bbs_drawing)