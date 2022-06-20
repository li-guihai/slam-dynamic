# 将Webots采集数据的真值轨迹转为kitti数据集格式

import numpy as np
import math
from scipy.spatial.transform import Rotation as R

xyzs = []
rpys = []
with open('/home/hai/Data_D/BaiduNetdiskDownload/mydata/miner1/values.txt', 'r') as f:
    count = 0
    for line in  f.readlines():
        count += 1

        line = line.strip('\n').split(' ')
        if len(line) == 6:
            #rpy
            if count%4 == 3:
                rpy = [float(line[1]) * 180 / math.pi, float(line[3]) * 180 / math.pi, float(line[5]) * 180 / math.pi]
                rpys.append(rpy)
            else:
                xyz = [float(line[1]), float(line[3]), float(line[5])]
                xyzs.append(xyz)

print(len(xyzs))
print(len(rpys))

xyz0 = xyzs[0]
rpy0 = rpys[0]
t = 0.0
with open('/home/hai/Data_D/BaiduNetdiskDownload/mydata/miner1/groundtruth.txt', 'a') as fw:
    for i in range(len(xyzs)):
        # 转到第一帧坐标下
        xyz = [xyzs[i][j] - xyz0[j] for j in range(len(xyz0))]
        rpy = [rpys[i][j] - rpy0[j] for j in range(len(rpy0))]

        # 欧拉角到四元数
        r = R.from_euler('xyz', rpy, degrees=True)
        qua = r.as_quat()
        # # 欧拉角到旋转矩阵
        # rm = r.as_matrix()
        fw.write('{} {} {} {} {} {} {} {}\n'.format(t, xyz[0], xyz[1], xyz[2],  \
                qua[0], qua[1], qua[2], qua[3]))
        t += 0.2 #手动设置帧率
    fw.close()
