from ex2_utils import *
# from ex1_utils import *

import matplotlib.pyplot as plt
import time
import cv2


def conv1Demo():
    inSignal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([0, 1, 0])
    myConv = conv1D(inSignal, kernel)
    label = "after Convolution with [ "
    for j in range(len(kernel)):
        label += str(kernel[j])
        label += " "
    label += "]: "
    print("inSignal:", inSignal)
    print(label, myConv)
    print("the right ans:", np.convolve(inSignal, kernel, 'full'))


def conv2Demo():
    file_path = "coins.jpg"
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    kernel = np.array([[5, 0, 5], [5, 5, 0], [1, 2, 1]])
    myImage = conv2D(img, kernel)
    cvImage = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].set_title("my image after conv2")
    ax[0][0].imshow(myImage, cmap='gray')
    ax[1][1].imshow(cvImage, cmap='gray')
    ax[1][1].set_title("cv image after conv2")
    ax[0][1].imshow(img, cmap='gray')
    ax[0][1].set_title("original image")
    difference = myImage - cvImage
    ax[1][0].set_title("difference")
    ax[1][0].imshow(difference, cmap='gray')
    plt.show()


def derivDemo():
    file_path = "beach.jpg"
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    direction, magnitude, lx, ly = convDerivative(img)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(ly, cmap='gray')
    ax[0][0].set_title("ly", fontdict=None, loc=None, pad=20, y=None)
    ax[1][1].imshow(lx, cmap='gray')
    ax[1][1].set_title("lx", fontdict=None, loc=None, pad=20, y=None)
    ax[0][1].set_title("magnitude", fontdict=None, loc=None, pad=20, y=None)
    ax[0][1].imshow(magnitude, cmap='gray')
    ax[1][0].set_title("direction", fontdict=None, loc=None, pad=20, y=None)
    ax[1][0].imshow(direction, cmap='gray')
    fig.tight_layout()
    plt.show()


def blurDemo():
    kernel = np.array([[2, 1, 1], [0, 1, 0], [2, 1, 0]])
    file_path = "boxman.jpg"
    in_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    blurImage = blurImage2(in_img, kernel.shape)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(in_img, cmap='gray')
    ax[0].set_title("original image", fontdict=None, loc=None, pad=20, y=None)
    ax[1].imshow(blurImage, cmap='gray')
    ax[1].set_title("image after gaussian filter", fontdict=None, loc=None, pad=20, y=None)
    plt.show()


def edgeDemo():
    file_path = "boxman.jpg"
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    # sobel:

    cvImage, myImage = edgeDetectionSobel(img)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(myImage, cmap='gray')
    ax[2].imshow(cvImage, cmap='gray')
    ax[0].set_title("original image")
    ax[1].set_title("my image after sobel")
    ax[2].set_title("cv image after sobel")
    plt.show()

    # LOG :

    fig, ax = plt.subplots(1, 2)
    edgeImage = edgeDetectionZeroCrossingLOG(img)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("original image")
    ax[1].imshow(edgeImage, cmap='gray')
    ax[1].set_title("image after log")
    plt.show()

    # Canny:

    cvImage, myImage = edgeDetectionCanny(img, 0.7, 0.3)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("original image")
    ax[1].imshow(myImage, cmap='gray')
    ax[1].set_title("my image after canny")
    ax[2].imshow(cvImage, cmap='gray')
    ax[2].set_title("cv image after canny")
    plt.show()


def houghDemo():
    file_path = "pool_balls.jpeg"
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    circle = houghCircle(img, 18, 20)
    draw(img, circle)


def draw(img: np.ndarray, ans: list):
    fig, ax = plt.subplots()
    for x, y, r in ans:
        circle = plt.Circle((x, y), r, color='black', fill=False)
        center = plt.Circle((x, y), 0.5, color='b')
        ax.add_patch(circle)
        ax.add_patch(center)
    ax.imshow(img)
    plt.show()


def main():
    print(myID())
    # convolution 1
    #conv1Demo()

    # convolution 2
    conv2Demo()

    # convDerivative
    # derivDemo()

    # Blurring
    # blurDemo()

    # edge detection
    # edgeDemo()

    # houghDemo
    # houghDemo()


if __name__ == '__main__':
    main()
