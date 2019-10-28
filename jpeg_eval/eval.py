from timeit import default_timer as timer
import math
import click
import numpy as np
import torchvision.models
from tqdm import tqdm
import pretrainedmodels
import pretrainedmodels.utils as pretrained_utils
import torch,torchvision
from torchvision import datasets,models,transforms
import os,sys
import time
from tqdm import tqdm
import argparse
torch.backends.cudnn.deterministic = True

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/data/zhijing/flickrImageNetV2/markov_cache/trial0/', type=str)
    parser.add_argument('--models', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    print(args)
    return args.dataset,args.models,args.batch_size
def run(args):
    dataset,models,batch_size = parse_args(args)
    all_models = ['alexnet',
              'densenet121',
              'densenet161',
              'densenet169',
              'densenet201',
              'inception_v3',
              'resnet101',
              'resnet152',
              'resnet18',
              'resnet34',
              'resnet50',
              'squeezenet1_0',
              'squeezenet1_1',
              'vgg11',
              'vgg11_bn',
              'vgg13',
              'vgg13_bn',
              'vgg16',
              'vgg16_bn',
              'vgg19',
              'vgg19_bn']

    extra_models = []
    for m in pretrainedmodels.model_names:
        if m not in all_models:
            all_models.append(m)
            extra_models.append(m)


    dataset_filename = dataset
    if models == 'all':
        models = all_models
    else:
        models = models.split(',')
    for model in models:
        assert model in all_models

    print('Reading dataset from {} ...'.format(dataset_filename))
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    image_datasets = datasets.ImageFolder(os.path.join(dataset_filename), data_transforms)
    data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,shuffle=True,num_workers=4)
    pt_model = getattr(torchvision.models, model)(pretrained=True)
    #if (torch.cuda.is_available()):
    pt_model = pt_model.cuda()
    pt_model.eval()
    #torch.backends.cudnn.benchmark = True
    num_images = 0
    num_top1_correct = 0
    num_top5_correct = 0
    predictions = []
    start = timer()
    with torch.no_grad():
        enumerable = enumerate(data_loader)
        total = int(math.ceil(len(data_loader) / batch_size))
        desc = 'Batch'
        enumerable = tqdm(enumerable, total=total, desc=desc)
        for ii, (img_input, target) in enumerable:
            img_input = img_input.cuda(non_blocking=True)
            _, output_index = pt_model(img_input).topk(k=5, dim=1, largest=True, sorted=True)
            output_index = output_index.cpu().numpy()
            predictions.append(output_index)
            for jj, correct_class in enumerate(target.cpu().numpy()):
                if correct_class == output_index[jj, 0]:
                    num_top1_correct += 1
                if correct_class in output_index[jj, :]:
                    num_top5_correct += 1
            num_images += len(target)
    end = timer()
    predictions = np.vstack(predictions)
    assert predictions.shape == (num_images, 5)
    top1_acc = num_top1_correct / num_images
    top5_acc = num_top5_correct / num_images
    total_time = end - start
    tqdm.write('    Evaluated {} images'.format(num_images))
    tqdm.write('    Top-1 accuracy: {:.2f}'.format(100.0 * top1_acc))
    tqdm.write('    Top-5 accuracy: {:.2f}'.format(100.0 * top5_acc))
    tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))

    return top1_acc,top5_acc

if __name__ == '__main__':
    run(sys.argv[1:])
