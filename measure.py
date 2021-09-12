import numpy as np
import glob
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


im_dir = 'results/predict/J/*.jpg'
label_dir = '../Dataset/UIE/UIEBD/test/label/'
p = 0
s = 0
k = 0
for item in sorted(glob.glob(im_dir)):

    k += 1
    name = item.split('/')[-1]
    im1 = Image.open(item).convert('RGB')
    im2 = Image.open(label_dir + name).convert('RGB')

    (h, w) = im2.size
    im1 = im1.resize((h, w))

    im1 = np.array(im1)
    im2 = np.array(im2)

    psnr_score = psnr(im1, im2)
    ssim_score = ssim(im1, im2, multichannel=True)
    print(item, ssim_score)
    p += psnr_score
    s += ssim_score
print(p/k)
print(s/k)





