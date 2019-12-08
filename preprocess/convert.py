from PIL import Image
import numpy as np
import os,sys
from torchvision import transforms
data_t = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])
def pic2bmp(file_in,file_out):
    img = Image.open(file_in)
    #file_out=file_out.replace('JPEG','bmp')
    data_t(img).convert('RGB').save(file_out)
#path = "/data/datasets/ILSVRC2012/val/"
#write_to = '/data/zhijing/ImageNet/bmps/'
path = '/data/zhijing/flickrImageNetV2/matched_frequency/'
#path = '/data/zhijing/flickrImageNetV2/topimages/'#topimages,threshold0.
path = '/data/zhijing/flickrImageNetV2/threshold0.7/'
#path = '/data/zhijing/flickrImageNetV2/markov_cache/trial3/dataset/'
write_to = '/data/zhijing/flickrImageNetV2/topimages0.7_224/'
if not os.path.exists(write_to):
    os.makedirs(write_to)

al=0
w = 0
h = 0
for d in os.listdir(path):
    l= len(os.listdir(os.path.join(path,d)))
    al += l
    if l!=10:
        print('wrong at')
        print(d, 'with number of ',l)
        #if not os.path.exists( os.path.join(write_to,d) ):
    #    os.makedirs(os.path.join(write_to,d))
    for f in os.listdir(os.path.join(path,d)):
        file_in = os.path.join(path,d,f)
    #    pic2bmp(file_in, os.path.join(write_to,d,f) )
        a, b = Image.open(file_in).size
        w+=a
        h+=b

print(w/10000,h/10000)
print(al)
