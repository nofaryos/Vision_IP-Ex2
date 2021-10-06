from cmath import cos
from math import pi, sin, cos

import cv2
import numpy as np


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208583476


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """

    flipKernel = kernel1[::-1]
    size_p = inSignal.size
    size_k = kernel1.size
    size_r = size_p + size_k - 1
    result = np.zeros(size_r)
    tempSignal = inSignal

    for i in range(size_k - 1):
        tempSignal = np.append(tempSignal, 0)
        tempSignal = np.insert(tempSignal, 0, 0)

    for i in range(size_r):
        result[i] = ((flipKernel * tempSignal[i:i + size_k]).sum())

    return result


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    1:return: The convolved image
    """
    kernel2 = np.fliplr(np.flipud(kernel2)).astype(np.int32)
    kernel2 = kernel2/kernel2.sum()
    out = np.zeros_like(inImage, dtype=np.int32)
    N, M = inImage.shape[:2]
    K, L = kernel2.shape[:2]

    K_half = K // 2
    L_half = L // 2
    pad_img = np.zeros((N + K_half * 2, M + L_half * 2), dtype=inImage.dtype)

    # pad top
    pad_img[0: K_half, L_half: M + L_half] = inImage[0]

    # pad bottom
    pad_img[N + K_half:, L_half: M + L_half] = inImage[-1]

    # pad left
    pad_img[K_half: N + K_half, 0: L_half] = inImage[:, 0].reshape(-1, 1)

    # pad right
    pad_img[K_half: N + K_half, M + L_half:] = inImage[:, -1].reshape(-1, 1)

    # top,left
    pad_img[0:K_half, 0:L_half] = inImage[0, 0]

    # top,right
    pad_img[0:K_half, M + L_half:] = inImage[0, -1]

    # bottom,left
    pad_img[N + K_half:, 0:L_half] = inImage[-1, 0]

    # bottom,right
    pad_img[N + K_half:, M + L_half:] = inImage[-1, -1]

    # fill image
    pad_img[K_half:N + K_half, L_half:M + L_half] = inImage

    print(inImage)
    print(pad_img)
    print('-------')
    for i in range(K_half, N + K_half):
        for j in range(L_half, M + L_half):
            out[i - K_half, j - L_half] = np.multiply(pad_img[i - K_half: i + K_half + 1, j - L_half: j + L_half + 1],
                                                      kernel2).sum()
            if kernel2.sum() != 0:
                out[i - K_half, j - L_half] /= kernel2.sum()
    return out.astype(inImage.dtype)


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """

    # calculate y_der
    kernelY = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    y_der = conv2D(inImage, kernelY)

    # calculate x_der
    kernelX = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    x_der = conv2D(inImage, kernelX)

    # calculate magnitude
    magnitude = pow(pow(x_der, 2) + pow(y_der, 2), 0.5)

    # calculate directions
    x_der = x_der + 0.000001
    # x_der[int(x_der) == 0] = 0.000001
    directions = np.arctan(y_der / x_der)

    return directions, magnitude, x_der, y_der


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    gaussianKer = cv2.getGaussianKernel(ksize=kernel_size[1], sigma=0)
    resultImg = cv2.filter2D(in_image, -1, gaussianKer)
    return resultImg


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """

    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx = Gx.astype(float) / 8
    Gy = Gy.astype(float) / 8

    smoothX = cv2.filter2D(img, -1, Gx, borderType=cv2.BORDER_REPLICATE)  # derivative - y, smooth - x
    smoothY = cv2.filter2D(img, -1, Gy, borderType=cv2.BORDER_REPLICATE)  # derivative - x, smooth - y

    myImage = pow(pow(smoothX, 2) + pow(smoothY, 2), 0.5)
    scale_factor = np.max(myImage) / 255
    myImage = (myImage / scale_factor).astype(float)

    img_cv = cv2.GaussianBlur(img, (3, 3), 0)
    sobel_x = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=3)  # derivative - y, smooth - x by cv
    sobel_y = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=3)  # derivative - x, smooth - y by cv

    cvImage = pow(pow(sobel_x, 2) + pow(sobel_y, 2), 0.5)
    myImage = threshFun(myImage, thresh)
    cvImage = threshFun(cvImage, thresh)
    print("cv", np.unique(cvImage))
    print("cv", np.unique(myImage))
    return cvImage, myImage


def threshFun(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if thresh * 255 <= img[i, j]:
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """

    # Smooth with 2D Gaussian
    smoothImg = cv2.GaussianBlur(img, (3, 3), 0)
    kernelLaplacian = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]])

    # Apply Laplacian filter
    laplacianImg = cv2.filter2D(smoothImg, -1, kernelLaplacian, borderType=cv2.BORDER_REPLICATE)
    rowsImg, colsImg = img.shape
    rowsDerive, colsDerive = laplacianImg.shape
    resultImg = np.zeros((rowsDerive, colsDerive))

    #  Look for patterns like {+,0,-} or {+,-} (zero
    # crossing)
    for i in range(1, rowsImg - 1):
        for j in range(1, colsImg - 1):
            neighbour = laplacianImg[i - 1: i + 2, j - 1: j + 2]
            M = neighbour.max()
            m = neighbour.min()
            p = laplacianImg[i, j]
            if p == 0.0 and m <= 0.0 <= M:
                resultImg[i, j] = 0
            elif m <= 0.0 <= p or p <= 0.0 <= M:
                resultImg[i, j] = 0
            else:
                resultImg[i, j] = 255
    return resultImg


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    # Smooth the image with a Gaussian
    smoothImg = cv2.GaussianBlur(img, (3, 3), 0)

    # Compute the partial derivatives Ix, Iy
    #  Compute magnitude and direction of the gradient:
    directions, magnitude, x_der, y_der = convDerivative(smoothImg)

    # from Radians to degrees
    direction = np.degrees(directions)

    scale_filter = magnitude.max()
    magnitude = (magnitude / scale_filter) * 255

    myImg = np.zeros_like(magnitude)

    rows, cols = img.shape

    # Quantize the gradient directions:
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            q = r = magnitude[row, col]

            if (0 <= direction[row, col] < 22.5) or (157.5 <= direction[row, col] <= 180):
                q = magnitude[row, col + 1]
                r = magnitude[row, col - 1]

            elif 22.5 <= direction[row, col] < 67.5:
                q = magnitude[row + 1, col - 1]
                r = magnitude[row - 1, col + 1]

            elif 67.5 <= direction[row, col] < 112.5:
                q = magnitude[row + 1, col]
                r = magnitude[row - 1, col]
            elif 112.5 <= direction[row, col] < 157.5:
                q = magnitude[row - 1, col - 1]
                r = magnitude[row + 1, col + 1]
            if (q < magnitude[row, col]) and (r < magnitude[row, col]):
                myImg[row, col] = magnitude[row, col]
            else:
                myImg[row, col] = 0

    # Hysteresis
    T1 = thrs_1 * 255
    T2 = thrs_2 * 255

    cvImage = cv2.Canny(smoothImg, T1, T2)

    sharpEdges = np.zeros((myImg.shape[0], myImg.shape[1]))

    for row in range(rows):
        for col in range(cols):
            if T1 < myImg[row, col]:
                sharpEdges[row, col] = myImg[row, col]

    for row in range(rows):
        for col in range(cols):
            if myImg[row, col] <= T2:
                myImg[row, col] = 0
            if T2 < myImg[row, col] <= T1:
                if sharpEdges[row - 1, col] == sharpEdges[row - 1, col + 1] == sharpEdges[row, col - 1] == sharpEdges[
                    row, col + 1] == \
                        sharpEdges[row + 1, col] == 0:
                    myImg[row, col] = 0

    return cvImage, myImg


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    cannyImage = cv2.Canny(img, 100, 200)

    rows, cols = cannyImage.shape
    edges = []
    points = []
    circlesResult = []
    helpCircles = {}
    ths = 0.47  # at least 0.47% of the pixels of a circle must be detected
    steps = 100

    for r in range(min_radius, max_radius + 1):
        for s in range(steps):
            angle = 2 * pi * s / steps
            x = int(r * cos(angle))
            y = int(r * sin(angle))
            points.append((x, y, r))

    for i in range(rows):
        for j in range(cols):
            if cannyImage[i, j] == 255:
                edges.append((i, j))

    for e1, e2 in edges:
        for d1, d2, r in points:
            a = e2 - d2
            b = e1 - d1
            s = helpCircles.get((a, b, r))
            if s is None:
                s = 0
            helpCircles[(a, b, r)] = s + 1

    sortedCircles = sorted(helpCircles.items(), key=lambda i: -i[1])
    for circle, s in sortedCircles:
        x, y, r = circle
        if s / steps >= ths and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circlesResult):
            print(s / steps, x, y, r)
            circlesResult.append((x, y, r))

    return circlesResult
