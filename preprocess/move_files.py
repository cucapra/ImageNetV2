import os,sys
import shutil
main_dir = '/data/zhijing/flickrImageNetV2/matched_frequency_224'
dir_list = os.listdir(main_dir)
new_dir = '/data/zhijing/flickrImageNetV2/matched_frequency_part4/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
for j,d in enumerate(os.listdir(main_dir)):
    #if j == 500:
    #    break
    copy_to=os.path.join(new_dir,d)
    if not os.path.exists(copy_to):
        os.makedirs(copy_to)
    for i,f in enumerate(os.listdir(os.path.join(main_dir,d)) ):
        #if i == 5:#%5 == 0:
        #    new_split = os.path.join(new_dir, 'bmp'+str(int(i/10) ))
        #    if not os.path.exists(new_split):
        #        os.makedirs(new_split)
        #    break
        copy_to = os.path.join(new_dir, d)
        if not os.path.exists(copy_to):
            os.makedirs(copy_to)
        #shutil.copy(os.path.join(main_dir,d,f),copy_to)
