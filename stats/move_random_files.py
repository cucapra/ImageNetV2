import os,sys
import shutil
import numpy as np
import random
#method = 'quality50'
method = 'sorted3697'
main_dirs = [
    #'/data/zhijing/flickrImageNetV2/cmp/matched/'+method,
    #'/data/zhijing/flickrImageNetV2/cmp/threshold0.7/'+method,
    '/data/zhijing/flickrImageNetV2/cmp/topimages/'+method
]

new_dirs_all = [
    #'/data/zhijing/flickrImageNetV2/stats2/matched/'+method,
    #'/data/zhijing/flickrImageNetV2/stats2/threshold0.7/'+method,
    '/data/zhijing/flickrImageNetV2/stats2/topimages/'+method
]
for main_dir, new_dir_all in zip(main_dirs, new_dirs_all):
    dir_list = os.listdir(main_dir)
    file_list = {x:os.listdir(os.path.join(main_dir, x)) for x in dir_list}
    
    for i in range(100):
        new_dir = os.path.join(new_dir_all, str(i))
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        pick = random.sample(list(range(10000)), k=1000)
        pick.sort()
        cnt = 0
        for j,d in enumerate(dir_list):
            copy_to=os.path.join(new_dir,d)
            if not os.path.exists(copy_to):
                os.makedirs(copy_to)
            if j == int(pick[cnt]/10) :
                for k, f in enumerate(file_list[d]):
                    if k==pick[cnt]%10:
                        shutil.copy(os.path.join(main_dir,d,f),copy_to)
                        cnt += 1
                        if cnt == 1000: break
                if cnt == 1000: break
