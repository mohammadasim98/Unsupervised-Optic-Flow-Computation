# -*- coding: utf-8 -*-
"""
@author: Mohammad Asim
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib.animation as animation
from IPython.display import display, HTML

def cvt_flow(item):
    """
    Convert optic flow component to a 3 channel RGB
    """
    
    mask = np.zeros((item.shape[0], item.shape[1], 3))
    mask[..., 1] = 255
    magnitude, angle = cv.cartToPolar(item[:, :, 0], item[:, :, 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    mask = np.array(mask, dtype=np.uint8)
    bgr = cv.cvtColor(mask,cv.COLOR_HSV2BGR)
    rgb2bgr = bgr[:, :, ::-1]
    
    return rgb2bgr

def rescale(img):
    img = img - img.min() # Now between 0 and 8674
    img = img / img.max() * 255
    
    return np.uint8(img)

def generate_video(flow, dataset, gt: bool=True):
    """
    Generate video from given set of predictions and the dataset used
    """
    
    rgb = []
    # Fetch ground truths from the dataset
    if gt:
        flow_gt_list = []
        for batch in dataset:       
            for fl in batch[1]:
                flow_gt_list.append(fl)
            
    for i in range(len(flow)):
        
        # Fetch optic flow components
        item = flow[i, :, :, 0:2]
        
        # Convert flow to rgb format
        flow_pred = cvt_flow(item)
        img = cv.cvtColor(flow[i, :, :, 2], cv.COLOR_GRAY2RGB)
        img = rescale(img)
        superposed = cv.addWeighted(flow_pred, 1, img, 1, 0)
        if(gt):
            flow_gt = cvt_flow(flow_gt_list[i].numpy())
            combined = cv.hconcat([flow_pred, flow_gt, img, superposed])
            
        else:
            combined = cv.hconcat([flow_pred, img, superposed])
            
        rgb.append(combined) 

    frame = 0
    dpi = 72
    xpixels, ypixels = rgb[0].shape[:2]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    im = plt.figimage(rgb[0])
    plt.figure(figsize=(20, 16))
    # # Opens a new window and displays the output frame
    plt.imshow(rgb[20]) 
    def animate(i):
        im.set_array(rgb[i])
        im.autoscale()
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(rgb), interval=33, repeat_delay=1, repeat=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('video.mp4', writer=writer)
