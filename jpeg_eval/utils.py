import numpy as np
import os
import sys
import csv
import cv2
import math

def diff_fit(acc1, rate):
    coef = np.array([-6.40782216e-05,-1.81228974e-03,6.46255250e-01])
    rates = np.array([rate**2,rate,1])
    return acc1 - np.sum(rates*coef)

def store_csv(row, name):
    with open(name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
def store_csv_check(row,name,line=['i','acc1','acc5','rate','cmp_mean','cmp_std']):
    if not os.path.isfile(name):
        store_csv(line,name)
    store_csv(row,name)

def psnr(img1, img2):
    original = cv2.imread(img1)
    contrast = cv2.imread(img2)
    
    mse = np.mean( (original - contrast) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def get_psnr(uncmp_path, cmp_path):
    psnr_mean = 0
    length = 0
    for dname in os.listdir(uncmp_path):
        fnames = os.listdir(os.path.join(uncmp_path,dname))
        for fname in fnames:
            uncmp_f = os.path.join(uncmp_path,dname,fname)
            cmp_f = os.path.join(cmp_path, dname, fname.replace(".bmp",".jpg"))
            psnr_mean += psnr(uncmp_f,cmp_f)
        length += len(fnames)
    return psnr_mean/length


def get_size(start_path):
    total_size = []
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size.append(os.path.getsize(fp))
    total_size = np.array(total_size)
    
    return total_size.mean(),total_size.std()

def create_dir(n):
    if not os.path.exists(n):
        os.makedirs(n)
def compress(dir_list,file_list,cmp_dir,uncmp_root,tmp_qtable):
    for dir_in in dir_list:
        if not os.path.exists(os.path.join(cmp_dir,dir_in) ):
            os.makedirs(os.path.join(cmp_dir,dir_in) )
        for file_in in file_list[dir_in]:
            file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            execute = "./cjpeg -outfile "+file_out+" -quality 50 -qtable "+tmp_qtable+" -qslots 0 "+os.path.join(uncmp_root,dir_in,file_in)
            os.system(execute)
def compress_quality(quality, dir_list, file_list, cmp_dir, uncmp_root):
    for dir_in in dir_list:
        if not os.path.exists(os.path.join(cmp_dir,dir_in) ):
            os.makedirs(os.path.join(cmp_dir,dir_in) )
        for file_in in file_list[dir_in]:
            file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            execute = "./cjpeg -outfile "+file_out+" -quality "+quality+" "+os.path.join(uncmp_root,dir_in,file_in)
            os.system(execute)

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]
