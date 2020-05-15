import multiprocessing
import pathlib
import tensorlayer as tl
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import matplotlib.pyplot as plt
import os
from typing import NewType

class DataLoader:
    def __init__(self,
                 FilePath: NewType('train data path',str),
                 FilePathL: NewType('label data path',str),
                 batch_size: NewType('batch_size',int)):
        
        self.train_data = self.produce_train(batch_size,FilePath)
        self.labels = self.produce_labels(batch_size,FilePathL)
        self.size = 0

    def get_train(self):
        return self.train_data

    def get_labels(self):
        return self.labels

    def __len__(self):
        if not self.size:
            raise NotImplementedError
        else:
            return self.size

    def produce_labels(self,batch_size,thePath):

        def generator():
            for img in labels:
                yield img

        DataList = tl.files.load_file_list(path=thePath, regx='.*.png', printable=False)
        labels = tl.vis.read_images(DataList, path=thePath, n_threads=32)
        labels = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
        labels = labels.batch(batch_size)
        labels = labels.prefetch(buffer_size=AUTOTUNE)
        return labels

    def produce_train(self,batch_size,thePath):

        def random_edit(img):
            hr_img = tf.image.random_flip_up_down(img)
            hr_img = tf.image.random_flip_left_right(hr_img)
            lr_img = tf.image.resize(hr_img, size=[48, 48])
            return lr_img/255.0, hr_img/255.0

        def generator():
            for img in train_data:
                img = tf.image.random_crop(img, [192, 192, 3])
            yield img

        for _ in os.listdir(thePath):
            self.size += 1

        DataList = tl.files.load_file_list(path=thePath, regx='.*.png', printable=False)
        train_data = tl.vis.read_images(DataList, path=thePath, n_threads=32)

        train_data = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
        train_data = train_data.map(random_edit, num_parallel_calls=multiprocessing.cpu_count())

        train_data = train_data.batch(batch_size)
        train_data = train_data.prefetch(buffer_size=AUTOTUNE)

        return train_data

    def show_example(self):
        for img in self.train_data:
            plt.imshow(img[0][1]/255.0)
            plt.imshow(img[0][0] / 255.0)
