from __future__ import print_function
import sys
import os
import cv2
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot,COCOroot
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300
from models.FSSD_VGG import build_net
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
from matplotlib import pyplot as plt

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def test_net(net,img,name,detector,transform,priors,top_k=200,thresh=0.01):

    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        x = x.cuda()
        scale = scale.cuda()

    out = net(x,test=True)
    boxes, scores = detector.forward(out, priors)
    boxes = boxes[0]
    scores = scores[0]

    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    flag = True
    for j in range(1, 21):
        inds = np.where(scores[:, j] > thresh)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = nms(c_dets, 0.45, force_cpu=True)
        c_dets = c_dets[keep, :]
        cls = np.ones(c_dets.shape[0])*j
        c_dets = np.column_stack((c_dets,cls))
        if flag:
            result = c_dets
            flag = False
        else:
            result = np.vstack((result,c_dets))

    reslut = list(result)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)
    currentAxis = plt.gca()

    for (x1,y1,x2,y2,s,cls) in result:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cls = int(cls)
        title = "%s:%.2f" % (CLASSES[int(cls)], s)
        coords = (x1,y1), x2-x1+1, y2-y1+1
        color = colors[cls]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(x1, y1, title, bbox={'facecolor': color, 'alpha': 0.5})
    plt.axis('off')
    plt.savefig(name.split('.')[0]+'.eps',format='eps',bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    Image = os.listdir('image/')
    for img_name in Image:
        img = cv2.imread("image/"+img_name)
        model = 'fssd_voc_79_74.pth'
        net = build_net(300, 21)
        state_dict = torch.load(model)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        net = net.cuda()
        cudnn.benchmark = True
        print("Finished loading model")
        transform = BaseTransform(300, (104, 117, 123), (2, 0, 1))
        detector = Detect(21, 0, VOC_300)
        priorbox = PriorBox(VOC_300)
        with torch.no_grad():
            priors = priorbox.forward()
            priors = priors.cuda()
        test_net(net, img, img_name, detector, transform, priors,top_k=200, thresh=0.4)
