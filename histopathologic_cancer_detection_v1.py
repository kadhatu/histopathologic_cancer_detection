# https://www.kaggle.com/c/histopathologic-cancer-detection
# This project prefer to be running in kaggle kernal

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# import modules
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook

# load data
data = pd.read_csv('/kaggle/input/train_labels.csv')
train_path = '/kaggle/input/train/'
test_path = '/kaggle/input/test/'

# quick look at the label stats
data['label'].value_counts()

# Plot some images with and without cancer tissue for comparison
def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img


# random sampling
# data comes from /kaggle/input/train_labels.csv
shuffled_data = shuffle(data)
fig, ax = plt.subplots(2,5,figsize=(20,8))
fig.suptitle('Histopathologic scans of lymph node sections', fontsize=20)

# Negatives
# enumerate type has its own index, so set i
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):
    # print (i, idx)
    # train_path = '/kaggle/input/train/'
    path = os.path.join(train_path, idx)
    ax[0,i].imshow(readImage(path + '.tif'))
    
    # Create a Rectangle patch
    # patches.Rectangle param : (x,y), width, height
    box = patches.Rectangle((32,32), 32, 32, linewidth=4, edgecolor='b',
                            facecolor='none',linestyle=':', capstyle='round')
    
    # add additional patches on image
    ax[0,i].add_patch(box)
ax[0,0].set_ylabel('Negative samples', size='large')

# Positives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[1,i].imshow(readImage(path + '.tif'))
    # Create a Rectangle path
    box = patches.Rectangle((32,32), 32, 32, linewidth=4, edgecolor='r',
                            facecolor='none',linestyle=':', capstyle='round')
    ax[1,i].add_patch(box)
ax[1,0].set_ylabel('Tumor tissue samples', size='large')


## Test function readCroppedImage() ##

# fig, ax = plt.subplots(1,2,figsize=(8,4))
# fig.suptitle('Test', fontsize=20)
# idx = shuffled_data['id'][0]
# path = os.path.join(train_path, idx)
# ax[0].imshow(readImage(path + '.tif'))
# ax[1].imshow(readCroppedImage(path + '.tif'))

# imagearray = readCroppedImage(path + '.tif', augmentations=False).reshape(-1,3)
# print(imagearray)


# Preprocessing and augmentation
# random rotation, random crop, random flip(horizontal and vertical both)
# random lighting, random zoom(not implemented here)
# Gaussian blur (noy implemented here), using OpenCV with image operations
# it seems faster than PIL or scikit-image

import random
ORIGINAL_SIZE = 96    # original size of the images - do not change

# AUGMENTATION VARIABLES
CROP_SIZE = 90    # final size after crop 

# range(0-180), 180 allows all rotation variations, 0=no change
RANDOM_ROTATION = 3
# center crop shift in x and y axes, 0=no change
RANDOM_SHIFT = 2
# range(0-100), 0=no change
RANDOM_BRIGHTNESS = 7
# range(0-100), 0=no change
RANDOM_CONTRAST = 5
# 0 or 1= random turn to left or right
RANDOM_90_DEG_TURN = 1


# augmentations parameter is included for counting statistics from images, 
# where we don't want augmentations
def readCroppedImage(path, augmentations=True):
    # OpenCV reads the image in bgr format by defualt
    bgr_img = cv2.imread(path)
    
    # We flip it to rgb for visualization purpose
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    
    if(not augmentations):
        return rgb_img / 255
    
    # random rotation
    rotation = random.randint(-RANDOM_ROTATION, RANDOM_ROTATION)
    if RANDOM_90_DEG_TURN == 1:
        rotation += random.randint(-1,1) * 90
    
    # (48,48) = (cols/2,rows/2) means the rotation center of the original images
    # M means 仿射变换, 2x3浮点矩阵 
    M = cv2.getRotationMatrix2D((48,48), rotation, 1)
    
    # (原图像, 2x3仿射变换矩阵, 生成图像的大小)
    rgb_img = cv2.warpAffine(rgb_img, M, (96,96))
    
    x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    
    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
    end_crop = start_crop + CROP_SIZE
    rgb_img = rgb_img[(start_crop+x):(end_crop+x), (start_crop+y):(end_crop+y)]/255

    # Random flip
    flip_hor = bool(random.getrandbits(1))
    flip_ver = bool(random.getrandbits(1))
    if flip_hor:
        rgb_img = rgb_img[:, ::-1]
    if flip_ver:
        rgb_img = rgb_img[::-1, :]
        
    # Random brightness
    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100
    rgb_img = rgb_img + br
    
    # Random contrast 
    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100
    rgb_img = rgb_img * cr
    
    # clip values to 0-1 range
    rgb_img = np.clip(rgb_img, 0, 1.0)
    
    return rgb_img


# Read cropped image for both negatives and positives
fig, ax = plt.subplots(2, 5, figsize=(20,8))
fig.suptitle('Cropped histopathologic scans of lymph node sections', fontsize=20)

# Negatives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[0,i].imshow(readCroppedImage(path + '.tif'))
ax[0,0].set_ylabel('Negative samples', size='large')

# Positives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[1,i].imshow(readCroppedImage(path + '.tif'))
ax[1,0].set_ylabel('Negative samples', size='large')


# As we count the statistics, we can check if there are any completely 
# black or white images
dark_th = 10 / 255
bright_th = 245 / 255

too_dark_idx = []
too_bright_idx = []

x_tot = np.zeros(3)
x2_tot = np.zeros(3)

counted_ones = 0
for i, idx in tqdm_notebook(enumerate(shuffled_data['id']), 'computing statistics...\
(220025 it total)'):
    path = os.path.join(train_path, idx)
    imagearray = readCroppedImage(path + '.tif', augmentations=False).reshape(-1,3)
    if imagearray.max() < dark_th:
        too_dark_idx.append(idx)
        continue
    if imagearray.min() > bright_th:
        too_bright_idx.append(idx)
        continue
    x_tot += imagearray.mean(axis=0)
    x2_tot += (imagearray**2).mean(axis=0)
    counted_ones +=1

channel_avr = x_tot / counted_ones
channel_std = np.sqrt(x2_tot / counted_ones - channel_avr**2)
channel_avr, channel_std

# print the result for extremely dark or bright
print('There was {0} extremely dark image'. format(len(too_dark_idx)))
print('There was {0} extremely bright image'. format(len(too_bright_idx)))

print('Dark one:')
print(too_dark_idx)
print('Bright one:')
print(too_bright_idx)


# split train and test dataset
from sklearn.model_selection import train_test_split
train_df = data.set_index('id')

train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)

tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)),
                               test_size=0.1, stratify=train_labels, random_state=123)


# fastai 1.0
from fastai import *
from fastai.vision import *
from torchvision.models import *

arch = densenet169
BATCH_SIZE = 128
sz = CROP_SIZE
MODEL_PATH = str(arch).split()[1]

# create dataframe for the fastai loader
train_dict = {'name':train_path + train_names, 'label':train_labels}
df = pd.DataFrame(data=train_dict)

# create test dataframe
# no label for test dataframe
test_names = []
for f in os.listdir(test_path):
    test_names.append(test_path + f)
df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])

'''
# uncommend the below segment of code when running
# Subclass ImageList to use our own image opening function
class MyImageItemList(ImageList):
	def open(self, fn:PathOrStr)->Image:
		img = readCroppedImage(fn.replace('/./','').replace('//','/'))
		return vision.Image(px=pil2tensor(img, np.float32))
'''
# Create ImageDataBunch using fastai data block API
imgDataBunch = (MyImageItemList.from_df(path='/', df=df, suffix='.tif')
               .split_by_idx(val_idx)
               .label_from_df(cols='label')
               .add_test(MyImageItemList.from_df(path='/', df=df_test))
                
                # We have our custom transformations implenmented in the image loader
                # but we could apply transformations also here.
                # Even thought we don't apply transformations here, we set two empty
                # lists to tfms. Train and Validation augmentations.
               .transform(tfms=[[],[]], size=sz)
               .databunch(bs=BATCH_SIZE)
                
                # Normalize with training set stats. These are means and std's of each
                # three channel and we calculated these previous in the stats step
               .normalize([tensor([0.70244707, 0.54624322, 0.69645334]),
                           tensor([0.23889325, 0.28209431, 0.21625058])])
               )

imgDataBunch.show_batch(rows=2, figsize=(4,4))


# Training
# We define a convnet learner object where we set the model architecture and our 
# data bunch. create_cnn changed to cnn_learner in fastai
# ps = dropout percentage(0-1) in the final layer

def getLearner():
    return cnn_learner(imgDataBunch, arch, pretrained=True, path='.',metrics=accuracy,
                     ps=0.5, callback_fns=ShowGraph)
learner = getLearner()


# 1cycle policy
# We can use lr_find with different weight decays and record all losses so that 
# we can plot them on the same graph
lrs = []
losses = []
wds = []
iter_count = 600

# WEIGHT DECAY = 1e-6
learner.lr_find(wd=1e-6, num_it=iter_count)
lrs.append(learner.recorder.lrs)
losses.append(learner.recorder.losses)
wds.append('1e-6')
learner = getLearner()

# WEIGHT DECAY = 1e-4
learner.lr_find(wd=1e-4, num_it=iter_count)
lrs.append(learner.recorder.lrs)
losses.append(learner.recorder.losses)
wds.append('1e-4')
learner = getLearner()

# WEIGHT DECAY = 1e-2
learner.lr_find(wd=1e-2, num_it=iter_count)
lrs.append(learner.recorder.lrs)
losses.append(learner.recorder.losses)
wds.append('1e-2')
learner = getLearner()


# Plot weight decays
_, ax = plt.subplots(1,1)
min_y = 0.5
max_y = 0.55
for i in range(len(losses)):
    ax.plot(lrs[i], losses[i])
    min_y = min(np.asarray(losses[i]).min(), min_y)
ax.set_ylabel('Loss')
ax.set_xlabel('Learning Rate')
ax.set_xscale('log')

ax.set_xlim((1e-3, 3e-1))
ax.set_ylim((min_y - 0.02, max_y))
ax.legend(wds)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

# try another learning rate and weight decays
max_lr = 2e-2
wd = 1e-4
learner.fit_one_cycle(cyc_len=8, max_lr=max_lr, wd=wd)

learner.recorder.plot_lr()
learner.recorder.plot_losses()

learner.fit_one_cycle(cyc_len=12, max_lr=slice(4e-5, 4e-4))
learner.recorder.plot_losses()


# Save the finetuned model
learner.save(MODEL_PATH + '_stage1')
# Load the baseline model
learner.load(MODEL_PATH + '_stage1')
# unfreeze and run learning rate finder again
learner.unfreeze()
learner.lr_find(wd=wd)
# plot learning rate finder results
learner.recorder.plot()


# Now, samller learning rates. This time we define the min and max lr of the cycle
learner.fit_one_cycle(cyc_len=12, max_lr=slice(4e-5, 4e-4))
learner.recorder.plot_losses()

# lets take a second look at the confusion matrix. See if how much we improved.
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(title='Confusion matrix')

# Save the finetuned model
learner.save(MODEL_PATHL_PATHL_PATHL_PATHL_PATH + '_stage2')


# Validation and analysis
preds, y, loss = learner.get_preds(with_loss=True)
acc = accuracy(preds, y)
print('The accuracy is {0} %.'.format(acc))


from random import randint
def plot_overview(inter:ClassificationInterpretation, classes=['Negative', 'Tumor']):
    # top losses will return all validation losses and indexes sorted by the
    # largest first.
    tl_val, tl_idx = interp.top_losses()
    # classes = interp.data.classes
    fig, ax = plt.subplots(3,4, figsize=(16,12))
    fig.suptitle('Predicted / Actual / Loss / Probability', fontsize=20)
    
    for i in range(4):
        random_index = randint(0, len(tl_idx))
        # tl means top loss
        idx = tl_idx[random_index]
        im, cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        im = image2np(im.data)
        cl = int(cl)
        ax[0,i].imshow(im)
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / \
        {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)
    
    
    for i in range(4):
        idx = tl_idx[i]
        im, cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        cl = int(cl)
        im = image2np(im.data)      
        ax[1,i].imshow(im)
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / \
        {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=8)
    
                       
    for i in range(4):
        idx = tl_idx[len(tl_idx) -i - 1]
        im, cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        cl = int(cl)
        im = image2np(im.data)      
        ax[2,i].imshow(im)
        ax[2,i].set_xticks([])
        ax[2,i].set_yticks([])
        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / \
        {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)

plot_overview(interp, ['Neigative', 'Tumor'])


# Gradient-weighted Class Activation Mapping (Grad-CAM)
# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
from fastai.callbacks.hooks import * 

# hook into forward pass
def hooked_backward(m, oneBatch, cat):
    # we hook into the convolutional part = m[0] of the model
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(oneBatch)
            preds[0,int(cat)].backward()
    return hook_a,hook_g


# We can create a utility function for getting a validation image with an 
# activation map.
def getHeatmap(val_index):
    """Returns the validation set image and the activation map"""
    m = learner.model.eval()
    tensorImg, cl = imgDataBunch.valid_ds[val_index]
    
    # create a batch from the one image
    oneBatch, _ = imgDataBunch.one_item(tensorImg)
    oneBatch_im = vision.Image(imgDataBunch.denorm(oneBatch)[0])
    
    # convert batch tensor image to grayscale image with opencv
    cvIm = cv2.cvtColor(image2np(oneBatch_im.data), cv2.COLOR_RGB2GRAY)
    
    # attach hooks
    hook_a, hook_g = hooked_backward(m, oneBatch, cl)
    # get convolutional activations and average from channels
    acts = hook_a.stored[0].cpu()
    # avg_acts = acts.mean(0)
    
    # Grad-CAM
    grad = hook_g.stored[0][0].cpu()
    grad_chan = grad.mean(1).mean(1)
    grad.shape, grad_chan.shape
    mult = (acts*grad_chan[...,None,None]).mean(0)
    return mult, cvIm


# modify our plotting func a bit
def plot_heatmap_overview(interp:ClassificationInterpretation, classes= \
                          ['Negative', 'Tumor']):
    # top losses will return all validation losses and indexes
    # sorted by the largest first
    tl_val, tl_idx = interp.top_losses()
    
    fig, ax = plt.subplots(3,4, figsize=(16,12))
    fig.suptitle('Grad-CAM\n Predicted / Actual / Loss /Probability', fontsize=20)
    
    # Random
    for i in range(4):
        random_index = randint(0, len(tl_idx))
        idx = tl_idx[random_index]
        act, im = getHeatmap(idx)
        H, W = im.shape
        _, cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        cl = int(cl)
        ax[0,i].imshow(im)
        ax[0,i],imshow(im, cmap=plt.cm.gray)
        ax[0,i].imshow(act, alpha=0.5, extenr=(0,H,W,0), 
                       interpolation='bilinear', cmap='inferno')
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / \
        {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    
    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)
    
    # Most incorrect or top losses
    for i in range(4):
        idx = tl_idx[len(tl_idx) -i - 1]
        act, im = getHeatmap(idx)
        H, W = im.shape
        _, cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        cl = int(cl)
        ax[1,i].imshow(im)
        ax[1,i].imshow(im, cmap=plt.cm.gray)
        ax[1,i].imshow(act, alpha=0.5, extent=(0,H,W,0),
              interpolation='bilinear', cmap='inferno')
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80)
    
    
    # Most correct or least losses
    for i in range(4):
        idx = tl_idx[len(tl_idx) - i - 1]
        act, im = getHeatmap(idx)
        H,W = im.shape
        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        cl = int(cl)
        ax[2,i].imshow(im)
        ax[2,i].imshow(im, cmap=plt.cm.gray)
        ax[2,i].imshow(act, alpha=0.5, extent=(0,H,W,0),
              interpolation='bilinear', cmap='inferno')
        ax[2,i].set_xticks([])
        ax[2,i].set_yticks([])
        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')
    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)

plot_heatmap_overview(interp, ['Negative','Tumor'])


# ROC curve and AUC
from sklearn.metrics import roc_curve, auc
probs = np.exp(preds[:, 1])

fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)

# Compute ROC area
roc_auc = auc(fpr, tpr)
print('ROC area is {0}'.format(roc_auc))

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver opearating characteristic')
plt.legend(loc='lower right')


# Submit predictions
learner.load(MODEL_PATH + '_stage1')

n_aug = 12
preds_n_avg = np.zeros((len(learner.data.test_ds.items),2))
for n in tqdm_notebook(range(n_aug), 'Running TTA'):
    preds, y = learner.get_preds(ds_type=DatasetType.Test, with_loss=False)
    preds_n_avg = np.sum([preds_n_avg, preds.numpy()], axis=0)
preds_n_avg = preds_n_avg / n_aug


# Next, we will transform class probabilities to just tumor class.probabilities
print('Negative and Tumor Probabilities:' + str(preds_n_avg[0]))
tumor_preds = preds_n_avg[:, 1]
print('Tumor probability:' + str(tumor_preds[0]))
class_preds = np.argmax(preds_n_avg, axis=1)
classes = ['Negative', 'Tumor']
print('Class prediction:' + classes[class_preds[0]])


# Submit the model for evaluation
# get test id's from the sample_submission.csv and keep their original order

SAMPLE_SUB = '/kaggle/input/sample_submission.csv'
sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = list(sample_df.id)


pred_list = [p for p in tumor_preds]
pred_dic = dict((key, value) for (key, value) in zip(learner.data.test_ds.items, pred_list))
pred_list_cor = [pred_dic['///kaggle/input/test/' + id + '.tif'] for id in sample_list]
df_sub = pd.DataFrame({'id':sample_list, 'label':pred_list_cor})
df_sub.to_csv('{0}_submission.csv'.format(MODEL_PATH), header=True, index=False)
df_sub.head(10)










