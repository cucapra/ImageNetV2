from PIL import Image
import numpy as np
jpg=Image.open('/data/datasets/ILSVRC2012/val/n13133613/ILSVRC2012_val_00019877.JPEG')
bmp = '1.bmp'
jpg.convert('RGB').save(bmp)
bmp=Image.open('1.bmp')
boo = np.equal(np.array(jpg), np.array(bmp))
print(boo.size-np.count_nonzero(boo))
