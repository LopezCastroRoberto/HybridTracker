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

# Select the player to track
def getPlayerStats(player, video, dir, jump, crop_x, crop_y, cropzx, cropzy, fasternet, ssdnet):
    file = open("player.txt", "w")
    file.write(player)
    file.close()
    # Select the video to evaluate the algorithm
    file = open("vidTrack", "w")
    file.write(dir+"/"+video)
    file.close()
    # Execute the tracker to get the results
    os.system('python hybrid_tracker.py ' + jump + " " + crop_x +
                    " " + crop_y + " " + fasternet + " " + ssdnet + " " + cropzx + " " + dir + " " + dir+"/"+video)
    # Get fps chart
    os.system('python fps_graph.py ' + dir + " " + jump + " " + crop_x +
                    " " + crop_y)
    # Accuracy charts
    os.system('python accuracy.py ' + dir + " " + jump + " " + crop_x +
                    " " + crop_y + " " + cropzx + " " + cropzy)

JUMP =  sys.argv[1]
CROP_X = sys.argv[2]
CROP_Y = sys.argv[3]

##### Kross results #####
CROP_Zx = '500'
CROP_Zy = '281'
getPlayerStats('Kross', '2_HQ2.mp4', 'paper_videos/kross_full/2_HQ2', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph_fasterrcnn_kross', 'inference_graph_ssd_kross')
getPlayerStats('Kross', '2_HQ8.mp4', 'paper_videos/kross_full/2_HQ8', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph_fasterrcnn_kross', 'inference_graph_ssd_kross')
getPlayerStats('Kross', '2_HQ15.mp4', 'paper_videos/kross_full/2_HQ15', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph_fasterrcnn_kross', 'inference_graph_ssd_kross')
##### Messi results #####
getPlayerStats('Messi', '4_HQ9.mp4', 'paper_videos/messi_full/4_HQ9', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph4', 'inference_graph2')
getPlayerStats('Messi', '4_HQ15.mp4', 'paper_videos/messi_full/4_HQ15', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph4', 'inference_graph2')
##### Modric results #####
CROP_Zx = '700'
CROP_Zy = '393'
getPlayerStats('Modric', '3_HQ3.mp4', 'paper_videos/modric_full/3_HQ3', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph_fasterrcnn_modric', 'inference_graph_ssd_modric')
getPlayerStats('Modric', '3_HQ8.mp4', 'paper_videos/modric_full/3_HQ8', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph_fasterrcnn_modric', 'inference_graph_ssd_modric')
getPlayerStats('Modric', '3_HQ19.mp4', 'paper_videos/modric_full/3_HQ19', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph_fasterrcnn_modric', 'inference_graph_ssd_modric')
##### Rakitic results #####
getPlayerStats('Rakitic', '1_HQ4.mp4', 'paper_videos/rakitic_full/1_HQ4', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph_fasterrcnn_rakitic', 'inference_graph_ssd_rakitic')
getPlayerStats('Rakitic', '1_HQ8.mp4', 'paper_videos/rakitic_full/1_HQ8', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph_fasterrcnn_rakitic', 'inference_graph_ssd_rakitic')
##### Messi results #####
getPlayerStats('Messi', '4_HQ16.mp4', 'paper_videos/messi_full/4_HQ16', JUMP, CROP_X, CROP_Y, CROP_Zx, CROP_Zy,
                'inference_graph4', 'inference_graph2')
