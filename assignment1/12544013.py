# Davi Fagundes Ferreira da Silva - 12544013
# SCC0251 - Prof Moacir Ponti - 7 semester

import numpy as np
import imageio.v3 as imageio

### Functions ###

def gamma_correction(img, gamma):
    return np.floor(255*((img.astype(np.int32)/255.0)**(1/gamma))).astype(np.uint8)

def histogram_equalization(img, hist):

    histC = np.zeros(256).astype(np.int32)

    # cumulative histogram
    histC[0] = hist[0]
    for i in range(1, 255):
        histC[i] = hist[i] + histC[i-1]
    
    hist_transf = np.zeros(256).astype(np.int32)

    N,M = img.shape

    img_eq = np.zeros([N,M]).astype(np.int32)

    for z in range(256):
        s = ( 255/float(np.sum(hist)) )*histC[z]
        hist_transf[z] = s

        img_eq[ np.where(img == z) ] = s

    return img_eq

def join_histograms(histograms):
    joint = np.zeros(histograms[0].shape).astype(np.int32)
    for hist in histograms:
        joint = joint + hist
    
    return joint

def superresolution(imgs):
    N, M = imgs[0].shape
    resulting_H = np.zeros([2*N, 2*M]).astype(np.uint8)

    for i in range(0,N):
        for j in range(0,M):
            resulting_H[2*i][2*j]     = imgs[0][i][j]
            resulting_H[2*i+1][2*j]   = imgs[1][i][j]
            resulting_H[2*i][2*j+1]   = imgs[2][i][j]
            resulting_H[2*i+1][2*j+1] = imgs[3][i][j]

    return resulting_H

def RMSE(H, H_hat):
    return np.sqrt(((H.astype(np.int32) - H_hat.astype(np.int32))**2).mean())

def main():
    file_prefix = input().rstrip()
    high_res_img_path = input().rstrip()
    high_img = imageio.imread(f"{high_res_img_path}")
    method = str(input().rstrip())
    gamma = float(input().rstrip())

    imgs = []
    for i in range(0,4):
        img = imageio.imread(f"{file_prefix}{i}.png")
        imgs.append(img)
    
    if method == '1': # Single Cumulative Histogram
        for i in range(len(imgs)):
            histogram, bin = np.histogram(imgs[i], bins=255)
            imgs[i] = histogram_equalization(imgs[i], histogram)

    elif method == '2': # Joint Cumulative Histogram
        histograms = []

        for img in imgs:
            histograms.append(np.histogram(img, bins = 255)[0])

        joint_histogram = join_histograms(histograms)

        for i in range(len(imgs)):
            imgs[i] = histogram_equalization(imgs[i], joint_histogram)

    elif method == '3': # Gamma Correction
        for i in range(len(imgs)):
            imgs[i] = gamma_correction(imgs[i], gamma)

    H_hat = superresolution(imgs)

    print(f"{RMSE(H=high_img, H_hat= H_hat):.4f}")
    
if __name__ == "__main__":
    main()

