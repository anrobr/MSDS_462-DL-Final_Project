import numpy as np
from collections import OrderedDict
from scipy.spatial import distance

class ObjectTracker():
    def __init__(self, maxFramesLost):
        self.counter = 0
        self.nextObjectId = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxFramesLost = maxFramesLost
        self.centroidsOnAppearance = OrderedDict()
        self.centroidsOnUpdate = OrderedDict()

    # bbs -> e.g. nms_bbs, where [center_x, center_y, ...]
    def update(self, bbs, frame_nr):
        if len(bbs) == 0:
            for objectID in list(self.disappeared.keys()): 
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxFramesLost:
                    self.deregister(objectID, frame_nr)
        else:
            # compute the bounding boxes centroids
            inputCentroids = np.zeros((len(bbs), 2), dtype=np.int)
            for (i, (cX, cY, _, _)) in enumerate(bbs):
                inputCentroids[i] = cX, cY

            if len(self.objects) == 0:
                for i in range(0, len(inputCentroids)):
                    self.register(inputCentroids[i], frame_nr)        
            else:
                objectIDs = list(self.objects.keys())
                objectCentroids = list(self.objects.values())
                # distance along the x-axis only                
                D = distance.cdist(np.array(objectCentroids), inputCentroids)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                usedRows = set()
                usedCols = set()

                for (row, col) in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    usedRows.add(row)
                    usedCols.add(col)

                unusedRows = set()
                unusedCols = set()

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(unusedCols)
            
                if D.shape[0] >= D.shape[1]:
                    for row in unusedRows:
                        objectID = objectIDs[row]
                        self.disappeared[objectID] += 1

                        if self.disappeared[objectID] > self.maxFramesLost:
                            self.deregister(objectID, frame_nr)
                else:
                    for col in unusedCols:
                        self.register(inputCentroids[col], frame_nr)
        
        return (self.counter, self.objects)

    def register(self, centroid, frame_nr):
        self.objects[self.nextObjectId] = centroid
        print(f'Registered new at Frame: {frame_nr} - Centroids for {self.nextObjectId}: {centroid}')
        # store first position on arrival of new object to later detect in which direction the object disappeared
        self.centroidsOnAppearance[self.nextObjectId] = centroid
        self.disappeared[self.nextObjectId] = 0
        self.nextObjectId += 1

    def deregister(self, objectID, frame_nr):
        print(f'Disposed old at Frame: {frame_nr} - UnusedRow object id: {objectID} - pos: {self.objects[objectID]} - started at: {self.centroidsOnAppearance[objectID]}')
        if abs(self.centroidsOnAppearance[objectID][0] - self.objects[objectID][0]) >= 50:
            if self.centroidsOnAppearance[objectID][0] - self.objects[objectID][0] > 0:
                self.counter += 1
            else: 
                self.counter -= 1
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.centroidsOnAppearance[objectID]