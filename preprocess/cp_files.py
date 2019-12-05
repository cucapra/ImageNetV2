import os,sys
import shutil
from PIL import Image
main_dir = '/data/datasets/ILSVRC2012/train'
dir_list = os.listdir(main_dir)
new_dir = '/data/zhijing/imagenet300/uncmp'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
for j,d in enumerate(os.listdir(main_dir)):
    #if j < 500:
    #    continue
    copy_to=os.path.join(new_dir,d)
    if not os.path.exists(copy_to):
        os.makedirs(copy_to)
    for i,f in enumerate(os.listdir(os.path.join(main_dir,d)) ):
        if i == 300:#%5 == 0:
        #    new_split = os.path.join(new_dir, 'bmp'+str(int(i/10) ))
        #    if not os.path.exists(new_split):
        #        os.makedirs(new_split)
            break
        if not os.path.exists(copy_to):
            os.makedirs(copy_to)
        img = Image.open(os.path.join(main_dir,d,f))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(os.path.join(copy_to, f.replace("JPEG", "bmp")))
        #shutil.copy(os.path.join(main_dir,d,f),copy_to)
