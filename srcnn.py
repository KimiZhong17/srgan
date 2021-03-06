import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d,UpSampling2d)
from tensorlayer.models import Model

class SRCNN(Model):
    def __init__(self):
        super(SRCNN,self).__init__()
        self.upsample1 = UpSampling2d(4, method='bicubic')
        w_init = tf.random_normal_initializer(stddev=0.02)
        self.conv1 = Conv2d(n_filter=32, filter_size=(9, 9), in_channels=3,act=tf.nn.relu, padding='SAME', W_init=w_init)
        self.conv2 = Conv2d(n_filter=16, filter_size=(1, 1),in_channels=32,act=tf.nn.relu, padding='SAME', W_init=w_init)
        self.conv3 = Conv2d(n_filter=3, filter_size=(5, 5),in_channels=16, padding='SAME', W_init=w_init)

    def forward(self,x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = tf.math.sigmoid(x)
        return x

if __name__ == "__main__":
    model = SRCNN()
    print(model)

