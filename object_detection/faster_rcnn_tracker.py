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
import time
import pandas as pd
import tensorflow as tf
import cv2

from utils import detector_utils as detector_utils
from utils import label_map_util

class FasterRCNN(multiprocessing.Process):
    def __init__(self, input_pipe, kcf_pipe, gpu_id, num_classes, jump, video_name, player, model_name):
        multiprocessing.Process.__init__(self)
        self.input_pipe = input_pipe
        self.kcf_pipe = kcf_pipe
        self.gpu_id = gpu_id
        self.num_classes = num_classes
        self.jump = jump
        self.video_name = video_name
        self.player = player
        self.model_name = model_name

    def run(self):
        cwd_path = os.getcwd()
        path_to_ckpt = os.path.join(cwd_path, self.model_name,'frozen_inference_graph.pb')
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

        print(detection_classes)

        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        video = cv2.VideoCapture(path_to_video)
        num_iter = 0
        
        while(video.isOpened()):

            _, frame = video.read()

            if not (num_iter % self.jump):
                if frame is None:
                    break
                frame_expanded = np.expand_dims(frame, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})

                (box, score) = self.best_score_box(boxes, scores, classes)

                # Send info to both threads
                self.input_pipe.send((num_iter, box, score))
                self.kcf_pipe.send((num_iter, box, score))                

            num_iter+=1

        return

    def best_score_box(self, boxes, scores, classes):
        pos_max = np.where(scores==np.amax(scores[np.where(classes==self.player)]))
        return boxes[pos_max], scores[pos_max]