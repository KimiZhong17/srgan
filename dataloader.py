import multiprocessing
import tensorlayer as tl
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import matplotlib.pyplot as plt
import os

class DataLoader:
    def __init__(self,sourcePath):
        self.sourcePath = sourcePath
        self.data = None
        self.size = None


    def __len__(self):
        if not self.size:
            self.size = 0
            for _ in os.listdir(self.sourcePath):
                self.size += 1

        return self.size


    def create_data(self):
        DataList = tl.files.load_file_list(path=self.sourcePath, regx='.*.png', printable=False)
        DataList = sorted(DataList)
        data = tl.vis.read_images(DataList, path=self.sourcePath, n_threads=32)
        return data
    
    def produce(self,batch_size):
        sources = self.create_data()

        def generator():          
            for s in sources:
                yield s
        
        def random_edit1(img):
            img = img / (255. / 2.)
            img = img - 1.
            source = tf.convert_to_tensor(img)
            return source

        def random_edit2(img):
            target = tf.image.random_crop(img, [384, 384, 3])
            target = target / (255. / 2.)
            target = target - 1.
            target = tf.image.random_flip_left_right(target)
            source = tf.image.resize(target, size=[96, 96])
            return source,target

        self.data = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
        if batch_size >1:
            self.data = self.data.map(random_edit2, num_parallel_calls=multiprocessing.cpu_count())
        else:
            self.data = self.data.map(random_edit1, num_parallel_calls=multiprocessing.cpu_count())
        self.data = self.data.batch(batch_size)
        self.data = self.data.prefetch(buffer_size=AUTOTUNE)

    def show_example(self):
        for img in self.data:
            plt.imshow(img[0]/255.0)
