import tensorflow as tf
import tensorlayer as tl
from srcnn import SRCNN
from dataloader import DataLoader
import matplotlib.pyplot as plt



def evaluate(name,
        evalloader,
        model,
        count):
    print(len(evalloader))
    evalloader.produce(batch_size = 1)
    writer = tf.summary.create_file_writer('./log/'+ name)
    for i,(X,Y) in enumerate(evalloader.train_data):

        model.eval()
        output = model(X)
        with writer.as_default():
            tf.summary.image('source_'+ str(i),X+1,step = 0)
            tf.summary.image('output_'+ str(i),output+1,step = 0)
            tf.summary.image('target_'+ str(i),Y+1,step = 0)


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
    evalloader = DataLoader('../srgan/DIV2K_valid_HR/')
    evaluate(name,evalloader,model_cnn,count)