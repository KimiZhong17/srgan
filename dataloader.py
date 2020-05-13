import multiprocessing
import pathlib
import tensorlayer as tl
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import matplotlib.pyplot as plt

class DataLoader:

    def __init__(self, FilePath):
        self.FilePath = FilePath
        self.ds = None
        self.train_data = None

    def get_ds(self,batch_size):

        def random_edit(img):
            hr_img = tf.image.random_flip_up_down(img)
            hr_img = tf.image.random_flip_left_right(hr_img)
            lr_img = tf.image.resize(hr_img, size=[96, 96])
            return lr_img, hr_img

        def generator():
            for img in self.ds:
                img = tf.image.random_crop(img, [384, 384, 3])
                yield img

        DataList = tl.files.load_file_list(path=self.FilePath, regx='.*.png', printable=False)
        self.ds = tl.vis.read_images(DataList, path=self.FilePath, n_threads=32)

        self.train_data = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
        self.train_data = self.train_data.map(random_edit, num_parallel_calls=multiprocessing.cpu_count())

        self.train_data = self.train_data.batch(batch_size)
        self.train_data = self.train_data.prefetch(buffer_size=AUTOTUNE)

    def show(self):
        for img in self.train_data:
            plt.imshow(img)

