import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)
from tensorlayer.models import Model

class SRCNN(Model):
    def __init__(self):
        super(CustomModel,self).__init__()
        
        self.conv1 = Conv2d(n_filter=64, filter_size=(9, 9),, padding='SAME')
        self.relu1 = tf.nn.relu()
        self.conv2 = Conv2d(n_filter=32, filter_size=(1, 1),, padding='SAME')
        self.relu2 = tf.nn.relu()
        self.conv3 = Conv2d(n_filter=1, filter_size=(5, 5),, padding='SAME')

    def forward(self,x,size):
        x = tl.prepro.imresize(x, size=size, interp='bicubic')
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)


        return x

 if __name__ == "__main__":
     model = SRCNN()
     print(model)

