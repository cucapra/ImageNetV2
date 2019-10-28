import os
path = '/data/zhijing/flickrImageNetV2/matched_frequency_part/'
original = '/data/zhijing/flickrImageNetV2/matched_frequency_224/'
dirs = os.listdir(original)
for i,d in enumerate(dirs):
    name = d.zfill(7)
    print(name)
    name = os.path.join(path,name)
    if not os.path.exists(name):
        os.makedirs(name)
        print(name)

