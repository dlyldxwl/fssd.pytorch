# Fssd.pytorch
A Pytorch0.4 re-implementation of Fssd detection network, with pretrained weights on VOC0712 and mAP=0.79.74.

Network | train data | test data | mAP | FPS | download
--|:--:|:--:|:--:|:--:|--:
FSSD | VOC 07+12 | VOC 07 test | 79.74 | 120 | [Baidu](https://pan.baidu.com/s/1m3i7gQGxZNk0cpqp4RBcXA) [Google](https://drive.google.com/file/d/1dpP2U6fWpb5CszwJS7q06A9gtX1fsBrS/view?usp=sharing)

Official caffe implementation is [here](https://github.com/lzx1413/CAFFE_SSD/tree/fssd), and pytorch0.3 re-implementation is [here](https://github.com/lzx1413/PytorchSSD). 

For details, please read the paper: [FSSD：Feature Fusion Single Shot Multibox Detector](https://arxiv.org/abs/1712.00960v1)
  
Our re-implementation is slightly better than paper(79.73 vs 78.8).  More high performance models will be available soon. 

## About more details
+ The implementation code abandons the bn layer after feature fusion.
+ Because the limitation of amount of gpu, batch_size is set to 32. If use 64 or more, I believe it will produce better performance
+ The code is mainly based on [rfbNet](https://github.com/ruinmessi/RFBNet). If you are interested in this project, please email me([yhao.chen0617@gmail.com](yhao.chen0617@gmail.com)) 

  
  
