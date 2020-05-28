import tensorflow as tf
import tensorlayer as tl
from srgan import get_G
from dataloader import DataLoader
import time



def train(name,
        trainloader,
        evalloader,
        model,
        n_epoch,
        learning_rate):

    optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
    writer = tf.summary.create_file_writer('./log/'+ name)
    train_weights = model.trainable_weights
    
    min_loss = float('inf')
    for epoch in range(n_epoch):
        
        ts = time.time()       
        loss = 0
        train_loss = 0
        
        for X,Y in trainloader.data:
            model.train()
            with tf.GradientTape() as tape:               
                output = model(X)
                loss = tl.cost.mean_squared_error(output,Y)
                train_loss += loss
            grad = tape.gradient(loss,train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))
            
        train_loss /= len(trainloader)
        
        
        eval_loss = 0
        for X,Y in evalloader.data:
            model.eval()
            with tf.GradientTape() as tape:               
                output = model(X)
                loss = tl.cost.mean_squared_error(output,Y)
                eval_loss += loss 
                
        eval_loss /= len(evalloader)

        time_cost = time.time() - ts
        
        print('iter %d, train_loss = %f , validation_loss = %f, time_cost = %f s'  %(epoch,train_loss,eval_loss,time_cost))
        
        if epoch >0:
            with writer.as_default():
                tf.summary.scalar('train_loss', train_loss, step=epoch+1)
                tf.summary.scalar('valid_loss', eval_loss, step=epoch+1)
                
        if eval_loss < min_loss:
            min_loss = eval_loss
            model.save_weights('./models/best_resnet_model.h5')

        

            












if __name__ == '__main__':

    ##############################################
    ##########      hyperparamter   ##############
    batch_size = 8
    learning_rate = 0.01
    n_epoch = 50
    name = 'srresnet_1'
    ##############################################
    trainloader = DataLoader('../srgan/DIV2K_train_HR/')
    trainloader.produce(batch_size)
    print('train data loaded')
    evalloader = DataLoader('../srgan/DIV2K_valid_HR/')
    evalloader.produce(batch_size)
    print('test data loaded')
    model = get_G()
    print(len(trainloader))
    
    train(name,trainloader,evalloader,model,n_epoch,learning_rate)