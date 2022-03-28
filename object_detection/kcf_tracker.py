#!/home/roberto/anaconda3/envs/tensorflow/bin/python

# Copyright 2022 Roberto Lopez Castro
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import multiprocessing
import time
import pandas as pd
import imutils
import cv2

from utils import detector_utils as detector_utils
from imutils.video import VideoStream
from imutils.video import FPS

class KCF(multiprocessing.Process):
    def __init__(self, tensor1, tensor2, jump, inst, video_name, width, path_to_results, crop_x):
        multiprocessing.Process.__init__(self)
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.jump = jump
        self.inst = inst
        self.video_name = video_name
        self.tracker = "kcf"
        self.width=width
        self.path_to_results=path_to_results
        self.crop_x=crop_x

    def center(self, x):
        x_c = x[0]+(x[2]-x[0])/2
        y_c = x[1]+(x[3]-x[1])/2
        return x_c, y_c

    def IoU(self, box, region, H, W):
        box = [box[0]*H, box[1]*W,box[2]*H, box[3]*W]

        xA = max(box[1], region[0])
        yA = max(box[0], region[1])
        xB = min(box[3], region[2])
        yB = min(box[2], region[3])

        interArea = max(0, xB-xA+1)*max(0, yB-yA+1)

        boxAArea = (box[3]-box[1]+1)*(box[2]-box[0]+1)
        boxBArea = (region[2]-region[0]+1)*(region[3]-region[1]+1)

        iou = interArea/float(boxAArea+boxBArea-interArea)
        return iou

    def run(self):
        (major, minor) = cv2.__version__.split(".")[:2]

        # For OpenCV 3.2 or before
        if int(major) == 3 and int(minor) < 3:
            tracker = cv2.Tracker_create(self.tracker.upper())

        # For OpenCV 3.3 or newer
        else:
            OPENCV_OBJECT_TRACKERS = {
                    #"csrt": cv2.TrackerCSRT_create,
                    "kcf": cv2.TrackerKCF_create,
                    "boosting": cv2.TrackerBoosting_create,
                    "mil": cv2.TrackerMIL_create,
                    "tld": cv2.TrackerTLD_create,
                    "medianflow": cv2.TrackerMedianFlow_create,
                    "mosse": cv2.TrackerMOSSE_create
            }

            # grab the appropriate object tracker using our dictionary of
            # OpenCV object tracker objects
            tracker = OPENCV_OBJECT_TRACKERS[self.tracker]()

        initBB = None
        cwd_path = os.getcwd()
        path_to_video = os.path.join(cwd_path,self.video_name)
        vs = cv2.VideoCapture(path_to_video)

        if self.inst:
            centers = ['x', 'y']
            c = pd.DataFrame([], columns=centers)
            fps_list = []

            columns_new = ['xmin_p', 'ymin_p', 'xmax_p', 'ymax_p', 'filename']
            df4 = pd.DataFrame([], columns=columns_new)

        # initialize the FPS throughput estimator
        counter=0
        acc=0
        redim=False
        x=None
        prevX=None
        times_diff=0
        incep=False
        last_ssd=[]
        last_kcf=[]

        # loop over frames from the video stream
        while True:
            if self.inst:
                start_detect = time.time()

            # grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture object
            frame = vs.read()
            frame = frame[1] #if args.get("video", False) else frame

            # check to see if we have reached the end of the stream
            if frame is None:
                break

            # resize the frame (so we can process it faster) and grab the
            # frame dimensions
            frame = imutils.resize(frame, width=self.width)
            (H, W) = frame.shape[:2]
            #(H, W) = (1080, 1920)

            if not (counter % self.jump):
                (n, box, score) = self.tensor1.recv()

                incep = True
                if x is not None:
                    iou = self.IoU(box[0], x, H, W)
                    cv2.putText(frame, "IoU: {:.4f}".format(iou), (10, 30),
                		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(box[0][1]*W), int(box[0][0]*H)),
                        (int(box[0][3]*W), int(box[0][2]*H)), (155, 155, 155), 2)
                    cv2.rectangle(frame, (int(x[0]), int(x[1])),
                        (int(x[2]), int(x[3])), (155, 155, 155), 2)

                    if iou > 0 or times_diff>4:
                        initBB = (int(box[0][1]*W), int(box[0][0]*H), int(box[0][3]*W-box[0][1]*W), int(box[0][2]*H-box[0][0]*H))
                        tracker.clear()
                        tracker = OPENCV_OBJECT_TRACKERS[self.tracker]()
                        tracker.init(frame, initBB)                        
                        x = (int(box[0][1]*W), int(box[0][0]*H), int(box[0][3]*W), int(box[0][2]*H))
                        times_diff=0
                    else:
                        times_diff+=1
                        incep = False
                else:
                    initBB = (int(box[0][1]*W), int(box[0][0]*H), int(box[0][3]*W-box[0][1]*W), int(box[0][2]*H-box[0][0]*H))
                    tracker.clear()
                    tracker = OPENCV_OBJECT_TRACKERS[self.tracker]()
                    tracker.init(frame, initBB)                    
                    x = (int(box[0][1]*W), int(box[0][0]*H), int(box[0][3]*W), int(box[0][2]*H))
                    times_diff=0

                if prevX is not None:
                    crop = prevX

                prevX = x

            # check to see if we are currently tracking an object
            #if initBB is not None:
            if not incep:
                # grab the new bounding box coordinates of the object
                (success, box) = tracker.update(frame)

                # check to see if the tracking was a success
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                            (0, 255, 0), 2)
                    x = (x,y,x+w,y+h)
                    last_kcf.append(x)

                if (counter % self.jump):
                    (box2, score) = self.tensor2.recv()
                else:
                    box2 = None

                if(box2 is not None):
                    acc += 1
                    cv2.rectangle(frame, (int(box2[0][1]*W), int(box2[0][0]*H)),
                        (int(box2[0][3]*W), int(box2[0][2]*H)), (255, 0, 0), 2)
                    last_ssd.append((int(box2[0][1]*W), int(box2[0][0]*H),
                        int(box2[0][3]*W), int(box2[0][2]*H)))

                if not success:
                    if(box2 is not None):
                        initBB = (int(box2[0][1]*W), int(box2[0][0]*H), int(box2[0][3]*W-box2[0][1]*W),
                            int(box2[0][2]*H-box2[0][0]*H))
                    else:
                        initBB = (prevBox[0]-5, prevBox[1]-5, prevBox[2]+5, prevBox[3]+5)
                        redim = True
                    x=(initBB[0], initBB[1], initBB[0]+initBB[2], initBB[1]+initBB[3])
                    tracker.clear()
                    tracker = OPENCV_OBJECT_TRACKERS[self.tracker]()
                    tracker.init(frame, initBB)
                    #fps = FPS().start()
                    last_kcf.append(x)
                elif redim:
                    redim = False
                    if box2 is not None:
                        iou = self.IoU(box2[0], x, H, W)
                        if iou>0:
                            initBB = (int(box2[0][1]*W), int(box2[0][0]*H), int(box2[0][3]*W-box2[0][1]*W), int(box2[0][2]*H-box2[0][0]*H))
                            tracker.clear()
                            tracker = OPENCV_OBJECT_TRACKERS[self.tracker]()
                            tracker.init(frame, initBB)
                            #fps = FPS().start()
                            x=(initBB[0], initBB[1], initBB[0]+initBB[2], initBB[1]+initBB[3])
                            last_kcf.append(x)
                prevBox = box                
            else:
                incep = False
                cv2.rectangle(frame, (int(x[0]), int(x[1])),
                    (int(x[2]), int(x[3])), (255, 0, 0), 2)

            if self.inst:
                fps_list.append((time.time()-start_detect)*1000)

                if x is not None:
                    columns_new = ["xmin_p", "ymin_p", "xmax_p", "ymax_p"]
                    df3 = pd.DataFrame([x], columns=columns_new)

                    df3 = df3.reindex(index=list(range(0, len(df3.index))), columns=list(df3.columns)+['filename'])
                    df3.loc[list(range(0, len(df3.index))), 'filename'] = ("training_"+str(counter)+".png")

                    frames = [df3, df4]
                    df4 = pd.concat(frames)

                    x_c, y_c = self.center(x)
                    columns_new = ["x", "y"]
                    df = pd.DataFrame([[x_c,y_c]], columns=columns_new)
                    c = pd.concat([df, c])
            X=None

            # show the output frame
            cv2.imshow("Frame", frame)
            #cv2.waitKey(0)
            key = cv2.waitKey(1) & 0xFF

            counter += 1

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        if self.inst:
            # if we are using a webcam, release the pointer
            #if not args.get("video", False):
            #    vs.stop()
            s = pd.Series(fps_list)
            #s.to_csv('fps.csv', index=False)
            s.to_csv(self.path_to_results + '/fps_'+str(self.jump)+'_'+str(self.crop_x)+'.csv')

            df4.to_csv(self.path_to_results + '/practical_labels_'+str(self.jump)+'_'+str(self.crop_x)+'.csv', index=False)

            c.to_csv('centers.csv', index=False)

        vs.release()

        # close all windows
        cv2.destroyAllWindows()        