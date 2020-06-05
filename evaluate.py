import tensorflow as tf
import tensorlayer as tl
from srcnn import SRCNN
from dataloader import DataLoader
import matplotlib.pyplot as plt
from srgan import get_G
from srresnet import SRresnet
import numpy as np

def evaluate(name,
        sourceloader,
        labelloader,
        model):

    sourceloader.produce(batch_size = 1)
    labelloader.produce(batch_size = 1)
    writer = tf.summary.create_file_writer('./log/'+ name)
    
    for i,(X,Y) in enumerate(zip(sourceloader.data,labelloader.data)):
        
            
        print('start')
        model.eval()
        output = model(X)
        
        X = (X+1.0)/2.0
        output = (output+1.0)/2.0
        Y = (Y+1.0)/2.0
        
        with writer.as_default():
            tf.summary.image('source_'+ str(i),X,step = 0)
            tf.summary.image('output_'+ str(i),output,step = 0)
            tf.summary.image('target_'+ str(i),Y,step = 0)
            print('saved')
            psnr = tf.image.psnr(output, Y, max_val=1.0)
            ssim2 = tf.image.ssim(output, Y, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
            mse = tf.keras.losses.mean_squared_error(output, Y).numpy()
#             print(psnr.numpy())
#             print(ssim2.numpy())
#             print(np.sum(mse)/(mse.size))
            print('psnr = %f, ssim2 = %f, mse = %f' %(psnr.numpy(),ssim2.numpy(),np.sum(mse)/(mse.size)))

        
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description= 'built-in bicubic method')
    parser.add_argument('-m', '--model', required=True, type=str, help="Model type")
    args = parser.parse_args()
    model_name = args.model
    if model_name == 'srcnn':
        model = SRCNN()
        model.load_weights('./models/best_cnn_model.h5')
    elif model_name == 'srgan':
        model = get_G([1, None, None, 3])
        model.load_weights('./models/g2.h5')
    elif model_name == 'srresnet':
        model = SRresnet()
        model.load_weights('./models/best_resnet_model.h5')
    else:
        print('invalid model')
        quit(1)
    sourceloader = DataLoader('../srgan/DIV2K_valid_LR_bicubic/X4/',ndata = 1)
    labelloader = DataLoader('../srgan/DIV2K_valid_HR/',ndata = 1)
    evaluate(model_name,sourceloader,labelloader,model)