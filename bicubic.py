from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

class Bicubic():
    def __init__(self):
        self.source = None
        self.target = None
        self.recovered = None
        self.mse = None

    def read_image(self,source,target=None):
        self.source = Image.open(source)
        if target:
            self.target =  Image.open(target)
        return self.source,self.target

    def show_image(self):
        if self.source:
            plt.figure('source')
            plt.imshow(np.array(self.source))
        if self.recovered:
            plt.figure('recovered')
            plt.imshow(np.array(self.recovered))
        if self.target:
            plt.figure('target')
            plt.imshow(np.array(self.target))
        plt.show()

    def interpolation(self,save_image_path = None):     
        self.recovered = self.source.resize(self.target.size,resample = Image.BICUBIC)
     
        if save_image_path:
            self.recovered.save(save_image_path)

        return self.recovered

    def cal_loss(self):
        temp = np.array(self.recovered)-np.array(self.target)
        self.mse = np.sum(temp*temp)/(np.array(self.recovered).size)
        return self.mse

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description= 'built-in bicubic method')
    parser.add_argument('-i', '--input_path', required=True, type=str, help="Input Image")
    parser.add_argument('-t', '--target_path', required=True, default= None, type=str, help="Target Image")
    parser.add_argument('-o', '--output_path', required=False, default= None, type=str, help="Output Image")
    args = parser.parse_args()

    model = Bicubic()
    model.read_image(args.input_path,args.target_path)
    model.interpolation(args.output_path)
    print('mse is ' + str(model.cal_loss()))
    model.show_image()




