import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)
from tensorlayer.models import Model

class SRresnet(Model):
    def __init__(self):
        super(SRresnet,self).__init__()
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)
        self.conv1 = Conv2d(n_filter=64,filter_size=(3, 3),strides=(1, 1),in_channels=3, act=tf.nn.relu, padding='SAME', W_init=w_init,b_init=None)
        self.conv2 = Conv2d(n_filter=64,filter_size=(3, 3),strides=(1, 1),in_channels=64, padding='SAME', W_init=w_init,b_init=None)
        self.conv3 = Conv2d(n_filter=256,filter_size=(3, 3),strides=(1, 1),in_channels=64, padding='SAME', W_init=w_init,b_init=None)
        self.conv4 = Conv2d(n_filter=3,filter_size=(3, 3),strides=(1, 1),in_channels=64, act=tf.nn.tanh, padding='SAME', W_init=w_init,b_init=None)
        self.bn2 = BatchNorm2d(num_features = 64,gamma_init=g_init)
        self.bn1 = BatchNorm2d(num_features = 64,gamma_init=g_init, act=tf.nn.relu)
        self.subconv1 = SubpixelConv2d(scale=2, n_out_channels=256,in_channels=256, act=tf.nn.relu)
        self.add1 = Elementwise(tf.add)

    def forward(self,x):
        x1 = self.conv1(x)
        x = x1
        for i in range(16):
            x2 = self.conv2(x)
            x2 = self.bn1(x2)
            x2 = self.conv2(x2)
            x2 = self.bn2(x2)
            x2 = self.add1([x,x2])
            x = x2
        
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.add1([x,x1])
        x = self.conv3(x)
        x = self.subconv1(x)
        x = self.conv3(x)
        x = self.subconv1(x)

        x = self.conv4(x)

        return x        


if __name__ == "__main__":
    model = SRresnet()
    print(model)

