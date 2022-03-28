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

import sys
import multiprocessing
from faster_rcnn_tracker import FasterRCNN
from ssd_tracker import SSD
from kcf_tracker import KCF

sys.path.append("..")

# General config vars
NUM_CLASSES=1
INST=1
PLAYER=1
JUMP=int(sys.argv[1])
CROP_X=int(sys.argv[2])
CROP_Y=int(sys.argv[3])
FASTER_MODEL_NAME= str(sys.argv[4])
SSD_MODEL_NAME=str(sys.argv[5])
RESIZE_WIDTH=int(sys.argv[6])
PATH_TO_RESULTS=sys.argv[7]
VIDEO_NAME=str(sys.argv[8])


if __name__ == "__main__":
    gpu_id=0 # More than one GPU can be added
    p_list=[]

    # A pipe for each consumer/client relation is created
    faster_ssd, ssd_faster = multiprocessing.Pipe()
    faster_kcf, kcf_faster = multiprocessing.Pipe()
    ssd_kcf, kcf_ssd = multiprocessing.Pipe()

    # Classes are created and added to the list of proccesses
    p = FasterRCNN(faster_ssd, faster_kcf, gpu_id, NUM_CLASSES, JUMP, VIDEO_NAME, PLAYER, FASTER_MODEL_NAME)
    p_list.append(p)

    p = SSD(ssd_faster, ssd_kcf, gpu_id, NUM_CLASSES, JUMP, CROP_X, CROP_Y, VIDEO_NAME, SSD_MODEL_NAME)
    p_list.append(p)

    p = KCF(kcf_faster, kcf_ssd, JUMP, INST, VIDEO_NAME, RESIZE_WIDTH, PATH_TO_RESULTS,CROP_X)
    p_list.append(p)

    # Starts each proccess's activity
    for p in p_list:
        p.start()

    # Main proccess until each child proccess terminates
    for p in p_list:
        p.join()
