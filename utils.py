import tensorflow as tf
def compare(path1,path2):
    im1 = tf.decode_png('path/to/im1.png')
    im2 = tf.decode_png('path/to/im2.png')
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    psnr = tf.image.psnr(im1, im2, max_val=1.0)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    mse = tf.keras.losses.mean_squared_error(im1, im2)
    return psnr, ssim2, mse




