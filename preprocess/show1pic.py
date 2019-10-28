import os
import sys

import matplotlib.pyplot as plt

repo_root = os.path.join(os.getcwd(), 'code')
sys.path.append(repo_root)

import candidate_data
import utils

cds = candidate_data.CandidateData(load_metadata_from_s3=False)
#tmpimg = cds.load_image('bbab60bda6162e17a84dcdb0b94d87fcbb228da5', size='original')
tmpimg = cds.load_image('f0ca9477a336a4e9ab5a7db0fc7221c0b4a68456', size='scaled_500')

plt.figure()
plt.imshow(tmpimg)
plt.show()
