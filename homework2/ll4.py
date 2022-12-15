from torch.nn import *
import torch.nn.functional as F


# 0.015
ll4 = []
ll = ll4
ll.append(BatchNorm2d(num_features=3))
ll.append(Conv2d(in_channels=3, out_channels=32,
          kernel_size=3, stride=1, padding=1))
ll.append(ReLU(inplace=True))
ll.append(Conv2d(in_channels=32, out_channels=32,
          kernel_size=3, stride=1, padding=1))
ll.append(ReLU(inplace=True))

ll.append(MaxPool2d(kernel_size=2, stride=2))

ll.append(BatchNorm2d(num_features=32))
ll.append(Conv2d(in_channels=32, out_channels=64,
          kernel_size=5, stride=1, padding=2))
ll.append(ReLU(inplace=True))
ll.append(Conv2d(in_channels=64, out_channels=64,
          kernel_size=5, stride=1, padding=2))
ll.append(ReLU(inplace=True))
ll.append(Conv2d(in_channels=64, out_channels=64,
          kernel_size=5, stride=1, padding=2))
ll.append(ReLU(inplace=True))

ll.append(MaxPool2d(kernel_size=2, stride=2))

ll.append(BatchNorm2d(num_features=64))
ll.append(Conv2d(in_channels=64, out_channels=64,
          kernel_size=5, stride=1, padding=2))
ll.append(ReLU(inplace=True))

ll.append(Conv2d(in_channels=64, out_channels=1, kernel_size=1))


class PlannerJT(torch.nn.Module):
    def __init__(self):

        super().__init__()

        # self.classifier = torch.nn.Linear(h, 2)
        # self.classifier = torch.nn.Conv2d(h, 1, 1)
        layer_list = []

        layer_list = ll4

        self.layers = torch.nn.Sequential(*layer_list)

        # (0): Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        # (1): ReLU(inplace=True)
        # (2): Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        # (3): ReLU(inplace=True)
        # (4): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        # (5): ReLU(inplace=True)
        # (6): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        # (7): ReLU(inplace=True)
        # (8): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, image_batch):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2) 
        """

        # print(image_batch.shape) # [batch_size,channel_size=3,height=96,width=128]

        x = self.layers(image_batch)

        # print(x.shape) #layers 的输出 [batch_size, 1, 6, 8]，channel 为 1
        # print(x[:,0].shape)

        # 輸入是[batch_size,height=6,width=8],輸出是[batch_size,2]
        return spatial_argmax_jt(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))
