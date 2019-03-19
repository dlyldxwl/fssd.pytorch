# Fssd.pytorch
A Pytorch0.4 re-implementation of Fssd detection network, with pretrained weights on VOC0712 and mAP=79.74.

Network | train data | test data | mAP | FPS | Download Link
--|:--:|:--:|:--:|:--:|--:
FSSD | VOC 07+12 | VOC 07 test | 79.74 | 90 | [Baidu](https://pan.baidu.com/s/1m3i7gQGxZNk0cpqp4RBcXA)/[Google](https://drive.google.com/file/d/1dpP2U6fWpb5CszwJS7q06A9gtX1fsBrS/view?usp=sharing)

Official caffe implementation is [here](https://github.com/lzx1413/CAFFE_SSD/tree/fssd), and pytorch0.3 re-implementation is [here](https://github.com/lzx1413/PytorchSSD). 

For details, please read the paper: [FSSD：Feature Fusion Single Shot Multibox Detector](https://arxiv.org/abs/1712.00960v1)
  
Our re-implementation is slightly better than paper (**79.74 vs 78.8**). Inference time is tested on **TITAN X Pasal and cudnn V5**. More high performance models will be available soon. 

## About more details
+ The implementation code abandons the bn layer after feature fusion.
+ Because the limitation of amount of gpu, batch_size is set to 32. If use 64 or more, I believe it will produce better performance.
+ If you want to re-produce more accurate model, I believe that **adding BN to the body of FSSD and applying Mixup**  will be very effective.
+ The code is mainly based on [rfbNet](https://github.com/ruinmessi/RFBNet). If you are interested in this project, please email me([yhao.chen0617@gmail.com](yhao.chen0617@gmail.com)) 

## Update

I add a prune file for FSSD, which bases on Network Slimming (ICCV 2017), for details, please refer to the paper: [Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html
).

 State | mAP | FPS | GPU | Model Size |Download Link
--|:--:|:--:|:--:|:--:|--:
Original |79.74 | 90 | TITAN X Pascal | 136M | [Baidu](https://pan.baidu.com/s/1m3i7gQGxZNk0cpqp4RBcXA)/[Google](https://drive.google.com/file/d/1dpP2U6fWpb5CszwJS7q06A9gtX1fsBrS/view?usp=sharing) 
First Prune (50%) | 79.64 | **150** | TITAN X Pascal | **52M** |[Google](https://drive.google.com/file/d/1RjQbZxwGepqaTeACVu8bdQiOY0HftWIb/view?usp=sharing) 
Second Prune  |  |  |  |  |

**Steps for pruning the model:**
+ Training the FSSD with BN model, meanwhile applying L1 constraints to the parameters of BN
+ Applying Network Slimming to the model trained in the previous step （Prune threshold is defaulted to 50%）
+ Re-train the pruned model, approximate 200 epochs, drop 0.1% in accuracy.
