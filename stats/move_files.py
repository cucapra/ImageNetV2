import os,sys
import shutil
import numpy as np
import random
#method = 'quality50'
methods = ['bound589', 'bys132','mab203']
for method in methods:
    main_dirs = [
        '/data/zhijing/flickrImageNetV2/cmp/matched/'+method,
        '/data/zhijing/flickrImageNetV2/cmp/threshold0.7/'+method,
        '/data/zhijing/flickrImageNetV2/cmp/topimages/'+method,
        '/data/zhijing/flickrImageNetV2/cmp/imgnet_val/'+method 
    ]
    
    new_dirs_all = [
        '/data/zhijing/flickrImageNetV2/stats2/matched/'+method,
        '/data/zhijing/flickrImageNetV2/stats2/threshold0.7/'+method,
        '/data/zhijing/flickrImageNetV2/stats2/topimages/'+method,
        '/data/zhijing/flickrImageNetV2/stats2/imgnet_val/'+method
    ]
    for main_dir, new_dir_all in zip(main_dirs, new_dirs_all):
        dir_list = os.listdir(main_dir)
        file_list = {x:os.listdir(os.path.join(main_dir, x)) for x in dir_list}
         
        for i in range(100):
            new_dir = os.path.join(new_dir_all, str(i))
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            pick_dir = random.sample(dir_list, k=700)
            for d in dir_list:
                cnt = 0
                copy_to=os.path.join(new_dir,d)
                if not os.path.exists(copy_to):
                    os.makedirs(copy_to)
                if d in pick_dir:
                    pick_pic = random.sample(file_list[d],k=3)
                    for f in pick_pic:
                        shutil.copy(os.path.join(main_dir,d,f),copy_to)
                        cnt += 1
                    #print(cnt)
