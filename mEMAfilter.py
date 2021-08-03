# %%
import numpy as np
from skimage import io
from skimage import data
from skimage import color
from matplotlib import pyplot as plt
# %%
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
# %%
# my_image = data.rocket()
my_image = io.imread('C:/Users/tanma/Desktop/video_test_pattern.jpg')
print(my_image.shape)
grayscale = color.rgb2gray(my_image)
show_image(grayscale, "Grayscale")
ycbcr = color.rgb2ycbcr(my_image)
show_image(ycbcr, "YCbCr")
 # %%
def EMAfilter(image, const=1):
    leftImage = np.zeros(image.shape)
    rightImage = np.zeros(image.shape)
    upImage = np.zeros(image.shape)
    downImage = np.zeros(image.shape)

    period=1
    sweeps = 4
    if const !=0:
        alpha = 1 - np.exp(-period/const)
    else:
        alpha = 1

    print('alpha: ', alpha)
    row_len, col_len = image.shape
    leftImage[:,0] = image[:,0]
    rightImage[:,-1] = image[:,-1]
    upImage[0,:] = image[0,:]
    downImage[-1,:] = image[-1,:]

    for col in range(col_len-1):
        leftImage[:, col+1] += ((1-alpha)*leftImage[:,col]+alpha*image[:,col+1])
        rightImage[:, col_len-col-2] += ((1-alpha)*rightImage[:,col_len-col-1]+alpha*image[:,col_len-col-2])
    
    for row in range(row_len-1):
        upImage[row+1, :] += ((1-alpha)*upImage[row, :]+alpha*image[row+1, :])
        downImage[row_len-row-2, :] += ((1-alpha)*downImage[row_len-row-1, :]+alpha*image[row_len-row-2, :])

    newImage = (leftImage+rightImage+upImage+downImage)/sweeps

    # show_image(newImage, 'EMA Filter')
    return newImage
# %%
newImage = EMAfilter(grayscale, const=20)
show_image(grayscale, 'Original')
show_image(newImage, 'Blurred')
# %%
newImage=np.zeros(my_image.shape)
for i in range(3):
    newImage[:,:,i]=EMAfilter(my_image[:,:,i], 10)

newImage = newImage.astype(int)
show_image(newImage, 'Color funked')
show_image(my_image, 'Original')
# %%
newImage = ycbcr
show_image(ycbcr, 'Original')
channels = [0]
for i in channels:
    print(i)
    newImage[:,:,i]=EMAfilter(ycbcr[:,:,i], 10)

newImage = newImage.astype(int)
show_image(newImage, 'Level Funked')
# %%
