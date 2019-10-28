import os,sys,shutil
import json,pathlib
import click
import numpy as np
import xml.etree.ElementTree as ET
import flickrapi
import urllib.request
import ssl,io
from PIL import Image
import imageio
repo_root = os.path.join(os.getcwd(),'../code')
sys.path.append(repo_root)
import candidate_data
import eval_utils
import imagenet

def fetchdata(url, count=1):
    try: 
        gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
        img_bytes = urllib.request.urlopen(url, context=gcontext).read()
        return img_bytes
    except:
        if count > 5: 
            print('Exceeds upper limits of trials to fetch data!')
            sys.exit() 
        fetchdata(url, count+1); 
  



@click.command()
@click.option('--dataset', default='imagenetv2-b-33')
@click.option('--store_root', default='/data/zhijing/flickrImageNetV2/matched_frequency')
def eval(dataset,store_root):
    flickr = flickrapi.FlickrAPI('f9916e41718d884f77d3f1941f83a242', '40e3e0dca45c841e', format='etree')
    
    dataset_filename = dataset
    dataset_filepath = pathlib.Path(__file__).parent / '../data/datasets' / (dataset_filename + '.json')
    print('Reading dataset from {} ...'.format(dataset_filepath))
    with open(dataset_filepath, 'r') as f:
        dataset = json.load(f)
    cur_imgs = [x[0] for x in dataset['image_filenames']]
    cds = candidate_data.CandidateData(load_metadata_from_s3=False, exclude_blacklisted_candidates=False,cache_on_local_disk=False)
    imgnet = imagenet.ImageNetData()
    print(len(cur_imgs))
    for i,img in enumerate(cur_imgs):
        if i < 9742: continue
        print(i)
        cd = cds.all_candidates[img]
        try:
            res=flickr.photos.getSizes(photo_id=cd['id_search_engine'])
            url = ""
            maxh = 0
            for j,child in enumerate(res[0]):
                ET.dump(child)
                sys.exit()
                h = int(child.attrib['height'])
                if child.attrib['media']=='photo' \
                   and h > maxh:
                   maxh = h
                   url = child.attrib['source']

            #url = (res[0][-1].attrib)['source']
            print(url)
            img_bytes=fetchdata(url)
            output = io.BytesIO(img_bytes)
            pil_image = Image.open(output) 
            pil_image.convert('RGB').save("1.bmp")
            pil_image.convert('RGB').save(dataset_filename+".bmp")
        except flickrapi.exceptions.FlickrError:
            print('Not found at',i,'th:',cd['id_search_engine'])
            with open(dataset_filename+'.json', 'a',newline='\n') as fp:
                json.dump(cd, fp)
            tmpimg=cds.load_image(cd['id_ours'], size='scaled_500')
            #print(type(tmpimg))
            #tmpimg.savefig('other.bmp')
            imageio.imwrite('1.bmp',tmpimg) 
            imageio.imwrite(dataset_filename+'.bmp',tmpimg) 
        data_loader = eval_utils.get_data_loader([img],imgnet,cds,image_size='scaled_500',resize_size=256,center_crop_size=224,batch_size=1)
        for _,img_class in data_loader:
            d = os.path.join(store_root,str(img_class.item()) )
            if not os.path.exists(d):
                os.makedirs(d)
            shutil.copyfile(dataset_filename+'.bmp',os.path.join(d,cd['id_ours']+'.bmp'))

if __name__ == '__main__':
    eval()
