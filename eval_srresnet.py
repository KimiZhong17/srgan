import tensorlayer as tl
from srresnet import get_D,get_G
import numpy as np

def evaluation(hr_img_path,lr_img_path):
    valid_hr_img_list = sorted(tl.files.load_file_list(path=hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=lr_img_path, regx='.*.png', printable=False))
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=lr_img_path, n_threads=32)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=hr_img_path, n_threads=32)

    imid = 0  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())


    G = get_G([1, None, None, 3])
    G.load_weights('./models/g.h5')
    G.eval()

    # valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    # valid_lr_img = valid_lr_img[np.newaxis, :, :, :]
    # size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = G(valid_lr_img).numpy()

    #print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], './saves/valid_gen.png')
    tl.vis.save_image(valid_lr_img[0], './saves/valid_lr.png')
    tl.vis.save_image(valid_hr_img, './saves/valid_hr.png')