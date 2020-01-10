import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def to_bmp(in_root, in_dirs, file_list, out_dir):
    for dir_in in in_dirs:
        temp_path = os.path.join(out_dir,dir_in)
        if not os.path.exists(temp_path) and os.path.isdir(os.path.join(in_root,dir_in)):
            os.makedirs(temp_path)
        if dir_in in file_list:
            cnt = 0
            for file_in in file_list[dir_in]:
                out_name = file_in.split('.')[0]+'.bmp'
                file_out = os.path.join(temp_path,out_name)
                if not os.path.isfile(file_out):
                    print(file_in, file_out)
                    try:
                        Image.open(os.path.join(in_root,dir_in,file_in)).convert('RGB').save(file_out)
                    except OSError as e:
                        print(e)
                        continue
                cnt += 1
                #if cnt == 10: break
# convert jpeg images to bmp
dir_list = os.listdir('/data/datasets/ILSVRC2012/val')
file_list = {}
for x in dir_list:
    temp_dir = os.path.join('/data/datasets/ILSVRC2012/val',x)
    if os.path.isdir(temp_dir):
        file_list[x] = os.listdir(temp_dir)
to_bmp('/data/datasets/ILSVRC2012/val', dir_list, file_list, '/data/zhijing/cs6787/imagenet/val_all/')

