import tensorflow as tf
import tensorlayer as tl
from srcnn import SRCNN
from dataloader import DataLoader
import matplotlib.pyplot as plt



def evaluate(name,
        dataloader,
        model,
        count):
    
    model.get_ds(1)
    writer = tf.summary.create_file_writer('./log/'+ name)
    for i,(X,Y) in enumerate(dataloader):

        model.eval()
        output = model(X)
        writer.add_figure('source_'+ str(i),X.eval())
        writer.add_figure('source_'+ str(i),output.eval())
        writer.add_figure('source_'+ str(i),Y.eval())


        if i>count:
            break

        
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    ##############################################
    ##########      hyperparamter   ##############
    count = 1
    name = 'srcnn_1'
    ##############################################
    
    model_cnn = SRCNN()
    model_cnn.load_weights('./models/best_cnn_model.h5')
    dataloader = DataLoader('../srgan/DIV2K_valid_HR/')
    evaluate(name,dataloader,model_cnn,count)