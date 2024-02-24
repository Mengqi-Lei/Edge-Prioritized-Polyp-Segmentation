# 实验记录
### 1. exp1
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 500
    lr = 1e-4
    early_stopping_patience = 50
    alpha=0.5   #控制edge损失权重
    beta=1    #控制互信息损失权重
#### 结果：
    Valid F1 improved from 0.8872 to 0.8882. Saving checkpoint: myfiles/debug/Kvasir-SEG/checkpoint.pth
    Epoch: 71 | Epoch Time: 0m 20s
	Train Loss: 0.8256 - Jaccard: 0.9174 - F1: 0.9532 - Recall: 0.9471 - Precision: 0.9676
    Val. Loss: 0.9852 - Jaccard: 0.8220 - F1: 0.8882 - Recall: 0.8956 - Precision: 0.9149

