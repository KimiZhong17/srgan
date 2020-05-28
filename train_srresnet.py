import tensorflow as tf
import tensorlayer as tl
from dataloader import DataLoader
import time
from srresnet import get_D,get_G

def train(
    name,
    trainloader,
    D,
    G,
    batch_size,
    n_epoch,
    learning_rate,
    tds=None):
    n_step_epoch = round(n_epoch // batch_size)
    G = get_G()
    D = get_D()
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')
    g_optimizer_init = tf.optimizers.Adam(learning_rate)
    g_optimizer = tf.optimizers.Adam(learning_rate)
    d_optimizer = tf.optimizers.Adam(learning_rate)
    G.train()
    D.train()
    VGG.train()

    for epoch in range(n_epoch):
        start_time = time.time()
        for step, (lr_patchs, hr_patchs) in enumerate(trainloader.data):
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                # compute loss and update model
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch, step, n_step_epoch, time.time() - start_time, mse_loss))

    for epoch in range(n_epoch):
        for step, (lr_patchs, hr_patchs) in enumerate(trainloader.data):
            if lr_patchs.shape[0] != batch_size:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                feature_fake = VGG((fake_patchs + 1) / 2.)  # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_patchs + 1) / 2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print(
                "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                    epoch, n_epoch, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss,
                    d_loss))

    tl.files.exists_or_mkdir('models')
    G.save_weights('./models/g.h5')
    D.save_weights('./models/d.h5')


if __name__ == '__main__':

    ##############################################
    ##########      hyperparamter   ##############
    batch_size = 8
    learning_rate = 0.01
    n_epoch = 50
    name = 'srres_1'
    ##############################################
    trainloader = DataLoader('../srgan/DIV2K_train_HR/')
    trainloader.produce(batch_size)
    print('train data loaded')
    D = get_D()
    G = get_G()
    print(len(trainloader))
    
    train(name,trainloader,D,G,batch_size,n_epoch,learning_rate)

