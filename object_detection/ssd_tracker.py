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
import numpy as np
import sys
import multiprocessing
import tensorflow as tf
import cv2

from utils import detector_utils as detector_utils
from utils import label_map_util
from utils import visualization_utils as vis_util

class SSD(multiprocessing.Process):
    def __init__(self, output_pipe, kcf_pipe, gpu_id, num_classes, jump, crop_x, crop_y, video_name, model_name):
        multiprocessing.Process.__init__(self)
        self.output_pipe = output_pipe
        self.kcf_pipe = kcf_pipe
        self.gpu_id = gpu_id
        self.num_classes = num_classes
        self.jump = jump
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.video_name = video_name
        self.model_name = model_name

    def run(self):
        cwd_path = os.getcwd()
        path_to_ckpt = os.path.join(cwd_path,self.model_name,'frozen_inference_graph.pb')
        path_to_labels = os.path.join(cwd_path,'training','labelmap.pbtxt')
        path_to_video = os.path.join(cwd_path,self.video_name)

        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.gpu_id)        

        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.4
            sess = tf.Session(config=config, graph=detection_graph)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        video=cv2.VideoCapture(path_to_video)
        
        counter=0
        iter=0
        num_frames=0
        color=(0, 0, 255)

        while(video.isOpened()):            
            if not (counter % self.jump):
                (n, box, score) = self.output_pipe.recv()
                _, frame = video.read()
                cv2.rectangle(frame, (int(box[0][1]*1920), int(box[0][0]*1080)),
                    (int(box[0][3]*1920), int(box[0][2]*1080)), color, 2)

            else:
                _, frame = video.read()
                try:
                    window, frame2 = self.frame_clipping(box, frame)
                    frame_expanded = np.expand_dims(frame2, axis=0)

                    iter+=1

                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: frame_expanded})
                    (box2, score) = self.best_score_box(boxes, scores)

                    # Only consider detection as valid, if score is greater than 75%
                    if score[0]>0.75:
                        box = self.resize_frame(window, box2, frame, frame2)
                        self.kcf_pipe.send((box, score))
                        color = (0, 255, 0)
                    else:
                        self.kcf_pipe.send((None, None))

                    for i in range(0, len(boxes[0])):
                        boxes[0][i] = self.resize_frame(window, [boxes[0][i]], frame, frame2)
                    
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.50)

                    cv2.rectangle(frame, (int(box[0][1]*1920), int(box[0][0]*1080)),
                        (int(box[0][3]*1920), int(box[0][2]*1080)), color, 2)
                    color = (0, 0, 255)
                except:
                    break

            num_frames += 1

            if cv2.waitKey(1) == ord('q'):                
                event.set()
                break
            counter+=1

        # Clean up
        video.release()
        self.output_pipe.close()
        cv2.destroyAllWindows()

    def best_score_box(self, boxes, scores):
        pos_max = np.where(scores==np.amax(scores))
        return np.array([boxes[pos_max][0]]), np.array([scores[pos_max][0]])

    def frame_clipping(self, box, frame):
        height, width, _ = frame.shape
        [ymin, xmin, ymax, xmax] = [box[0][0]*height-self.crop_y,
         box[0][1]*width-self.crop_x, box[0][2]*height+self.crop_y, box[0][3]*width+self.crop_x]

        ymin=int(round(ymin))
        ymax=int(round(ymax))
        xmin=int(round(xmin))
        xmax=int(round(xmax))

        ymin = 0 if ymin<0 else ymin
        ymax = height if ymax>height else ymax
        xmin = 0 if xmin<0 else xmin
        xmax = width if xmax>width else xmax

        return ([ymin, xmin], frame[ymin:ymax, xmin:xmax])


    def resize_frame(self, window, box, frame, frame2):
        height, width, _ = frame.shape
        height2, width2, _ = frame2.shape

        [ymin, xmin, ymax, xmax] = [window[0]+box[0][0]*height2,
         window[1]+box[0][1]*width2, window[0]+box[0][2]*height2, window[1]+box[0][3]*width2]

        ymin = 0 if ymin<0 else ymin
        ymax = height if ymax>height else ymax
        xmin = 0 if xmin<0 else xmin
        xmax = width if xmax>width else xmax

        return np.array([[ymin/height, xmin/width, ymax/height, xmax/width]])