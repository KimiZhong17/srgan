import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)
from tensorlayer.models import Model

def get_G(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02) # the initializer for weight matrix
    g_init = tf.random_normal_initializer(1., 0.02) # initializer for initializing gamma in batch normalization

    nin = Input(input_shape)
    n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init)(nin)
    temp = n

    # B residual blocks
    for i in range(16):
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
        nn = BatchNorm2d(gamma_init=g_init)(nn)
        nn = Elementwise(tf.add)([n, nn])
        n = nn

    n = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=g_init)(n)
    n = Elementwise(tf.add)([n, temp])
    # B residual blacks end

    n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

    n = Conv2d(256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

    nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init)(n)
    G = Model(inputs=nin, outputs=nn, name="generator")
    return G

def get_D(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02) # the initializer for weight matrix
    gamma_init = tf.random_normal_initializer(1., 0.02) # initializer for initializing gamma in batch normalization
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    nin = Input(input_shape)
    n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(nin)

    n = Conv2d(df_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=gamma_init)(n)
    n = Elementwise(combine_fn=tf.add, act=lrelu)([n, nn])

    n = Flatten()(n)
    no = Dense(n_units=1, W_init=w_init)(n)
    D = Model(inputs=nin, outputs=no, name="discriminator")
    return D
# class get_G(Model):
#     def __init__(self):
#         super(get_G,self).__init__()
#         w_init = tf.random_normal_initializer(stddev=0.02)
#         g_init = tf.random_normal_initializer(1., 0.02)
#         self.conv1 = Conv2d(n_filter=64,filter_size=(3, 3),strides=(1, 1),in_channels=3, act=tf.nn.relu, padding='SAME', W_init=w_init,b_init=None)
#         self.conv2 = Conv2d(n_filter=64,filter_size=(3, 3),strides=(1, 1),in_channels=64, padding='SAME', W_init=w_init,b_init=None)
#         self.conv3 = Conv2d(n_filter=256,filter_size=(3, 3),strides=(1, 1),in_channels=64, padding='SAME', W_init=w_init,b_init=None)
#         self.conv4 = Conv2d(n_filter=3,filter_size=(3, 3),strides=(1, 1),in_channels=64, act=tf.nn.tanh, padding='SAME', W_init=w_init,b_init=None)
#         self.bn2 = BatchNorm2d(num_features = 64,gamma_init=g_init)
#         self.bn1 = BatchNorm2d(num_features = 64,gamma_init=g_init, act=tf.nn.relu)
#         self.subconv1 = SubpixelConv2d(scale=2, n_out_channels=256,in_channels=256, act=tf.nn.relu)
#         self.add1 = Elementwise(tf.add)

#     def forward(self,x):
#         x1 = self.conv1(x)
#         x = x1
#         for i in range(16):
#             x2 = self.conv2(x)
#             x2 = self.bn1(x2)
#             x2 = self.conv2(x2)
#             x2 = self.bn2(x2)
#             x2 = self.add1([x,x2])
#             x = x2
        
#         x = self.conv2(x)
#         x = self.bn1(x)
#         x = self.add1([x,x1])
#         x = self.conv3(x)
#         x = self.subconv1(x)
#         x = self.conv3(x)
#         x = self.subconv1(x)

#         x = self.conv4(x)

#         return x 
        


# class get_D(Model):
#     def __init__(self):
#         super(get_D,self).__init__()
#         w_init = tf.random_normal_initializer(stddev=0.02) 
#         g_init = tf.random_normal_initializer(1., 0.02)
#         lrelu = lambda x: tl.act.lrelu(x, 0.2)
#         df_dim = 64
#         self.conv1 = Conv2d(n_filter=df_dim,filter_size=(4, 4),strides=(2, 2),in_channels=3, act=lrelu, padding='SAME', W_init=w_init)
#         self.conv2 = Conv2d(n_filter=df_dim * 2,filter_size=(4, 4),strides=(2, 2),in_channels=df_dim, padding='SAME', W_init=w_init, b_init=None)
#         self.bn1 =  BatchNorm2d(num_features = df_dim * 2, act=lrelu,gamma_init=g_init)
#         self.conv3 = Conv2d(n_filter=df_dim * 4,filter_size=(4, 4),strides=(2, 2),in_channels=df_dim* 2, padding='SAME', W_init=w_init, b_init=None)
#         self.bn2 =  BatchNorm2d(num_features = df_dim * 4, act=lrelu,gamma_init=g_init)
#         self.conv4 = Conv2d(n_filter=df_dim * 8,filter_size=(4, 4),strides=(2, 2),in_channels=df_dim* 4, padding='SAME', W_init=w_init, b_init=None)
#         self.bn3 =  BatchNorm2d(num_features = df_dim * 8, act=lrelu,gamma_init=g_init)
#         self.conv5 = Conv2d(n_filter=df_dim * 16,filter_size=(4, 4),strides=(2, 2),in_channels=df_dim* 8, padding='SAME', W_init=w_init, b_init=None)
#         self.bn4 =  BatchNorm2d(num_features = df_dim * 16, act=lrelu,gamma_init=g_init)
#         self.conv6 = Conv2d(n_filter=df_dim * 32,filter_size=(4, 4),strides=(2, 2),in_channels=df_dim* 16, padding='SAME', W_init=w_init, b_init=None)
#         self.bn5 =  BatchNorm2d(num_features = df_dim * 32, act=lrelu,gamma_init=g_init)
#         self.conv7 = Conv2d(n_filter=df_dim * 16,filter_size=(1, 1),strides=(1, 1),in_channels=df_dim* 32, padding='SAME', W_init=w_init, b_init=None)
#         self.bn6 =  BatchNorm2d(num_features = df_dim * 16, act=lrelu,gamma_init=g_init)
#         self.conv8 = Conv2d(n_filter=df_dim * 8,filter_size=(1, 1),strides=(1, 1),in_channels=df_dim* 16, padding='SAME', W_init=w_init, b_init=None)
#         self.bn7 =  BatchNorm2d(num_features = df_dim * 8, act=lrelu,gamma_init=g_init)
#         self.conv9 = Conv2d(n_filter=df_dim * 2,filter_size=(1, 1),strides=(1, 1),in_channels=df_dim* 8, padding='SAME', W_init=w_init, b_init=None)
#         self.bn8 =  BatchNorm2d(num_features = df_dim * 2, act=lrelu,gamma_init=g_init)
#         self.conv10 = Conv2d(n_filter=df_dim * 2,filter_size=(3, 3),strides=(1, 1),in_channels=df_dim* 2, padding='SAME', W_init=w_init, b_init=None)
#         self.bn9 =  BatchNorm2d(num_features = df_dim * 2, act=lrelu,gamma_init=g_init)
#         self.conv11 = Conv2d(n_filter=df_dim * 8,filter_size=(3, 3),strides=(1, 1),in_channels=df_dim* 2, padding='SAME', W_init=w_init, b_init=None)
#         self.bn10 =  BatchNorm2d(num_features = df_dim * 8,gamma_init=g_init)
#         self.add1 = Elementwise(combine_fn=tf.add, act=lrelu)
#         self.flat1 = Flatten()
#         self.dence1 = Dense(n_units=1, W_init=w_init, in_channels= 288*df_dim)



#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.bn1(x)
#         x = self.conv3(x)
#         x = self.bn2(x)
#         x = self.conv4(x)
#         x = self.bn3(x)
#         x = self.conv5(x)
#         x = self.bn4(x)
#         x = self.conv6(x)
#         x = self.bn5(x)
#         x = self.conv7(x)
#         x = self.bn6(x)
#         x = self.conv8(x)
#         x1 = self.bn7(x)
#         x = self.conv9(x1)
#         x = self.bn8(x)
#         x = self.conv10(x)
#         x = self.bn9(x)
#         x = self.conv11(x)
#         x = self.bn10(x)
#         x = self.add1([x,x1])
#         x = self.flat1(x)
#         x = self.dence1(x)

#         return x



if __name__ == "__main__":
    G = get_G()
    D = get_D()
    print(G)
    print(D)

