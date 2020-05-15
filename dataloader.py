import multiprocessing
import tensorlayer as tl
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import matplotlib.pyplot as plt
import os

class DataLoader:
    def __init__(self,FilePath,FilePathS,batch_size=1):
        self.size = None
        self.FilePath = FilePath
        self.train_data = None
        self.train_data_S = None
        self.labels = None
        self.hascrop = []
        self.produce_train(batch_size, FilePath)
        self.produce_train_S(batch_size, FilePathS)
        self.produce_labels(FilePath)


    def get_train(self):
        return self.train_data

    def get_train_S(self):
        return self.train_data_S

    def get_labels(self):
        return self.labels

    def get_train_P(self):
        return list(zip(self.get_train(), self.get_train_S()))


    def __len__(self):
        if self.size is None:
            self.size = 0
            for _ in os.listdir(self.FilePath):
                self.size += 1
        else:
            return self.size

    def produce_data(self,thePath):
        DataList = tl.files.load_file_list(path=thePath, regx='.*.png', printable=False)
        data = tl.vis.read_images(DataList, path=thePath, n_threads=32)
        return data

    def produce_labels(self,thePath):
        batch_size=1

        def random_edit(img):
            return img

        def generator():
            for img in data:
                img = tf.convert_to_tensor(img)
                yield img

        data = self.produce_data(thePath)
        self.labels = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
        self.labels = self.labels.map(random_edit, num_parallel_calls=multiprocessing.cpu_count())
        self.labels = self.labels.batch(batch_size)
        self.labels = self.labels.prefetch(buffer_size=AUTOTUNE)


    def produce_train(self,batch_size,thePath):

        def random_edit(img):
            # hr_img = tf.image.random_flip_up_down(img)
            # hr_img = tf.image.random_flip_left_right(hr_img)
            return img

        def generator():
            for img in data:
                img = tf.image.random_crop(img, [192, 192, 3])
                self.hascrop.append(img)
                yield img

        data = self.produce_data(thePath)
        self.train_data = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
        self.train_data = self.train_data.map(random_edit, num_parallel_calls=multiprocessing.cpu_count())
        self.train_data = self.train_data.batch(batch_size)
        self.train_data = self.train_data.prefetch(buffer_size=AUTOTUNE)


    def produce_train_S(self,batch_size,thePath):

        def random_edit(img):
            return img

        def generator():
            for img in self.hascrop:
                #img = tf.image.random_crop(img, [192, 192, 3])
                img = tf.image.resize(img, size=[48, 48])
                yield img

        self.train_data_S = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
        self.train_data_S = self.train_data_S.map(random_edit, num_parallel_calls=multiprocessing.cpu_count())
        self.train_data_S = self.train_data_S.batch(batch_size)
        self.train_data_S = self.train_data_S.prefetch(buffer_size=AUTOTUNE)

    def show_example(self):
        for img in self.train_data:
            plt.imshow(img[0]/255.0)
