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

import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import os
import cv2
import glob
import ntpath
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import spline
import sys

prt = pd.read_csv(sys.argv[1]+'/practical_labels_'+sys.argv[2]+'_'+sys.argv[3]+'.csv')
thc = pd.read_csv(sys.argv[1]+'/images_labels.csv')


def xcenter_t(x):
    return x['xmin_t']+(x['xmax_t']-x['xmin_t'])/2

def ycenter_t(x):
    return x['ymin_t']+(x['ymax_t']-x['ymin_t'])/2

def xcenter_p(x):
    return x['xmin_p']+(x['xmax_p']-x['xmin_p'])/2

def ycenter_p(x):
    return x['ymin_p']+(x['ymax_p']-x['ymin_p'])/2


prt['ymin_p'] = prt['ymin_p']*thc['height'][0]/int(sys.argv[6])
prt['xmin_p'] = prt['xmin_p']*thc['width'][0]/int(sys.argv[5])
prt['ymax_p'] = prt['ymax_p']*thc['height'][0]/int(sys.argv[6])
prt['xmax_p'] = prt['xmax_p']*thc['width'][0]/int(sys.argv[5])

center_x = prt.apply(xcenter_p, axis=1)
df1 = pd.DataFrame(center_x, columns=['x_center_p'])
center_y = prt.apply(ycenter_p, axis=1)
df2 = pd.DataFrame(center_y, columns=['y_center_p'])
prt = pd.concat([prt, df1, df2], axis=1)

center_x = thc.apply(xcenter_t, axis=1)
df1 = pd.DataFrame(center_x, columns=['x_center_t'])
center_y = thc.apply(ycenter_t, axis=1)
df2 = pd.DataFrame(center_y, columns=['y_center_t'])
thc = pd.concat([thc, df1, df2], axis=1)

total_frames = thc.shape[0]

general = pd.merge(prt, thc, how='right', on=['filename'])

def distance(x):
    return sqrt((x['x_center_p']-x['x_center_t'])**2 + (x['y_center_p']-x['y_center_t'])**2)

d = general.apply(distance, axis=1)
df2 = pd.DataFrame(d, columns=['distance'])
general = pd.concat([general, df2], axis=1)


general = general.dropna()
total_frames = general.shape[0]

result = general.loc[general.groupby("filename")["distance"].idxmin()]

def IoU(x):
    #determine de (x,y)-coordinates of the intersection rectangle
    xA = max(x['xmin_t'], x['xmin_p'])
    yA = max(x['ymin_t'], x['ymin_p'])
    xB = min(x['xmax_t'], x['xmax_p'])
    yB = min(x['ymax_t'], x['ymax_p'])

    #compute the area of the intersection rectangle
    interArea = max(0, xB-xA+1)*max(0, yB-yA+1)

    #compute the are of both the prediction and ground-truth rectangles
    boxAArea = (x['xmax_t']-x['xmin_t']+1)*(x['ymax_t']-x['ymin_t']+1)
    boxBArea = (x['xmax_p']-x['xmin_p']+1)*(x['ymax_p']-x['ymin_p']+1)

    #compute the intersection over union by tacking the intersection area and
    #dividing it by the sum of prediction + general_truth areas - the intersect
    #area
    iou = interArea/float(boxAArea+boxBArea-interArea)

    #return the intersection over union value
    return iou

result = result.apply(IoU, axis=1)
print("//////////")
general = (general.loc[result.index[:]]).sort_index()
print(general)
print("**********")
print(result.mean())
print("----------")
print(total_frames)
print(total_frames-result.shape[0])
print("----------")
print("..........")


x = list(range(0, (result.shape)[0]))
print(result.sort_index())
y = (result.sort_index()).tolist()
y.reverse()

fig = plt.figure()
plt.ylabel('average overlap', fontsize=20)
plt.xlabel('frames', fontsize=20)
plt.bar(x, y)
y = np.ones(len(y))*0.5
plt.plot(x, y, color=(1, 0, 0, 1))
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17, rotation=45)
plt.gcf().subplots_adjust(bottom=0.2, left=.15)
plt.savefig(sys.argv[1]+'/accy_'+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.png')

# ***** OPE *****
y_ope = [result.shape[0]/total_frames]
x_ope = [0]
for th in range(1, 11):
    y_ope.append((result[result >= (th/10)]).shape[0]/total_frames)
    x_ope.append(th/10)
    print(y_ope, " ", x_ope)
fig = plt.figure()

plt.xlabel('average overlap', fontsize=20)
plt.ylabel('correct frames', fontsize=20)
plt.grid(color='k', linestyle='-', linewidth=0.3)


plt.xticks(x_ope)
plt.yticks(np.linspace(0, 1, 11))
axes = plt.gca()
axes.set_ylim([-0.1,1.1])

x_sm = np.array(x_ope)
y_sm = np.array(y_ope)

xnew = np.linspace(x_sm.min(),x_sm.max(),100) #300 represents number of points to make between T.min and T.max

spl = make_interp_spline(x_sm, y_sm, k=1) #BSpline object
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth)
plt.legend(['AUC ' + "{0:.4f}".format(np.trapz(y_ope, x=x_ope))], prop={'size': 17})
plt.xticks(fontsize=15, rotation=30)
plt.yticks(fontsize=15, rotation=30)
plt.gcf().subplots_adjust(bottom=0.2, left=.15)
#plt.show()
plt.savefig(sys.argv[1]+'/ope_'+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.png')

print(np.trapz(y_ope, x=x_ope))