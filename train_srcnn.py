import tensorflow as tf
import tensorlayer as tl
from srcnn import SRCNN
from dataloader import DataLoader
import time




def train(name,
        trainloader,
        evalloader,
        model,
        optimizer,
        n_epoch,
        learning_rate,
        print_every,
        save_every):

    optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
    
    train_weights = model.trainable_weights
    for epoch in range(n_epoch):
        
        ts = time.time()
        model.train()
        loss = 0
        train_loss = 0

        for X,Y in trainloader.train_data:
            with tf.GradientTape() as tape:               
                output = model(X)
                loss = tl.cost.mean_squared_error(output,Y)
                train_loss += loss
            grad = tape.gradient(loss,train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))
            
        print(len(trainloader))
        train_loss /= len(trainloader)
        
        model.eval()
        eval_loss = 0
        for X,Y in evalloader.train_data:
            with tf.GradientTape() as tape:               
                output = model(X)
                loss = tl.cost.mean_squared_error(output,Y)
                eval_loss += loss      
        eval_loss /= len(evalloader)

        time_cost = time.time() - ts
        
        print('iter %d, train_loss = %f , validation_loss = %f, time_cost = %f'  %(epoch,train_loss,eval_loss,time_cost))


            












if __name__ == '__main__':

    ##############################################
    ##########      hyperparamter   ##############
    batch_size = 8
    learning_rate = 0.1
    n_epoch = 50
    print_every = 5
    save_every = 10
    name = 'srcnn_1'
    ##############################################
    trainloader = DataLoader('../srgan/DIV2K_train_HR/')
    trainloader.get_ds(batch_size)
    print('train data loaded')
    evalloader = DataLoader('../srgan/DIV2K_valid_HR/')
    evalloader.get_ds(batch_size)
    print('test data loaded')
    model = SRCNN()

    
    train(name,trainloader,evalloader,model,batch_size,n_epoch,learning_rate,print_every,save_every)


