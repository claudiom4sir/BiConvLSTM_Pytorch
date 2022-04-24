# BiConvLSTM_Pytorch
 Implementation of Bidirectional ConvLSTM in Pytorch
 
 The code from https://github.com/KimUyen/ConvLSTM-Pytorch has been modified to support multiple recurrent layers.
 
 ## Basic usage
 ```
 import torch.nn as nn
 import torch
 from biconvlstm import ConvLSTM
 
 input = torch.rand(2, 5, 3, 64, 64).cuda() # (B, T, C, H, W)
 gt = torch.rand(2, 5, 64, 64, 64).cuda()
 
 convlstm = ConvLSTM((64,64), return_sequence=False, layer_num=2, bidirectional=True).cuda()
 output, _, _ = convlstm(input)
 
 criterion = nn.MSELoss().cuda()
 loss = criterion(gt, output)
 loss.backward()
 ```
