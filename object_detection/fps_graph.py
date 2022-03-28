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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

fps = pd.read_csv(sys.argv[1]+'/fps_'+sys.argv[2]+'_'+sys.argv[3]+'.csv')

print(fps)
print(fps.ix[:,1].mean())
print(1000/fps.ix[2:,1].mean())

x = (fps.ix[2:,0]).tolist()
y = (fps.ix[2:,1]).tolist()

fig = plt.figure()
axes = plt.gca()
axes.set_ylim([0,70])
plt.ylabel('time (ms)', fontsize=20)
plt.xlabel('frames', fontsize=20)
plt.bar(x, y)

y = np.ones(len(y))*16
plt.plot(x, y, 'r')
y2 = np.ones(len(y))*20
plt.plot(x, y2, 'b')
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17, rotation=45)
plt.gcf().subplots_adjust(bottom=0.2, left=.15)
#plt.show()
plt.savefig(sys.argv[1]+'/fps_'+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+'.png')
