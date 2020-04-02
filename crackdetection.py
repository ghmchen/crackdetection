import cv2
import numpy as np
import array
import os
import csv
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

def extract_bv_ori(image):
    b, green_fundus, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv2.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    # vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000: # and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if (shape == "circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)

    # fig, axs =plt.subplots(1,2,figsize=(18,5))
    # axs[0].imshow(fundus_eroded,cmap='gray')
    # axs[1].imshow(blood_vessels, cmap='gray')
    # plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.1, rect=None)
    # plt.show()
    return blood_vessels

def crack_detection(image,use_bilateralFilter=True,use_gauss=True,use_erode3=True,use_cut200=True,check_shape=True):
    b, g, r = cv2.split(image)
    if use_bilateralFilter==True:
        image_bf = cv2.bilateralFilter(g, 5, 40, 40)  # bif1 = cv2.adaptiveBilateralFilter(image, 5, 40, 40)
    else:
        image_bf =g
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_bf_clahe = clahe.apply(image_bf)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(img_bf_clahe, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    f2 = cv2.subtract(R2, img_bf_clahe)

    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f3 = cv2.subtract(R3, img_bf_clahe)

    # r4 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)), iterations=1)
    # R4 = cv2.morphologyEx(r4, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)), iterations=1)
    # f4 = cv2.subtract(R4, img_bf_clahe)
    if use_gauss==True:
        kernel = cv2.getGaussianKernel(9, 1.8)
        window = np.outer(kernel, kernel.transpose())
        gaussianf = cv2.filter2D(f3, -1, window)
    else:
        gaussianf=f3
    f5 = clahe.apply(gaussianf)
    """improve threadshold-based segmtation"""

    """removing very small contours, which have small contourArea"""
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 300: #200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    if use_cut200==True:
        im_sub_area200 = cv2.bitwise_and(f5, f5, mask=mask)
        ret, im_sub_area200_binary = cv2.threshold(im_sub_area200, 15, 255, cv2.THRESH_BINARY_INV)
    else:
        im_sub_area200_binary=f6

    if use_erode3==True:
        im_sub_area200_binary_erode3 = cv2.erode(im_sub_area200_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        """decrease background size, it is to increase crack size"""
    else:
        im_sub_area200_binary_erode3=im_sub_area200_binary

    """ removing small blobs"""
    im_sub_area200_binary_erode3_not = cv2.bitwise_not(im_sub_area200_binary_erode3 ) #crack 1, background 0
    result=0
    if check_shape==True:
        xmask1 = np.ones(image.shape[:2], dtype="uint8") * 255
        xcontours, xhierarchy = cv2.findContours(im_sub_area200_binary_erode3_not, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contor in xcontours:
            if cv2.contourArea(contor) <= 1000:
                cv2.drawContours(xmask1, [contor], -1, 0, -1)

        finimage = cv2.bitwise_and(im_sub_area200_binary_erode3_not, im_sub_area200_binary_erode3_not, mask=xmask1)
        blood_vessels = cv2.bitwise_not(finimage)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))  # (12:width direction, 9 height direction)
        opening = cv2.morphologyEx(blood_vessels, cv2.MORPH_OPEN, kernel)
        """smoothing cracks"""

        result=blood_vessels
    else:
        result = im_sub_area200_binary_erode3

    # fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    # axs[0, 0].imshow(f5, cmap='gray')
    # axs[0, 1].imshow(im_sub_area200, cmap='gray')
    # axs[1, 0].imshow(im_sub_area200_binary_erode3_not, cmap='gray')
    # axs[1, 1].imshow(result, cmap='gray')
    # plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.1, rect=None)
    # plt.show()

    return  result

if __name__ == "__main__":

    is3D_raw=False
    if is3D_raw:
        filePathC = r"D:\data\E00000353.raw"
        count = int(os.stat(filePathC).st_size / 2)
        with open(filePathC, 'rb') as fp:
            binaryC3 = array.array("h")
            binaryC3.fromfile(fp, count)
        image = np.reshape(binaryC3, (512,1024))
        image = image.astype('float32')
        image *= 255.0/image.max()
        g= image.astype(np.uint8)

    filepath = r"D:\data\inputs\E3D00000076.jpg"
    input_img = cv2.imread(filepath)
    results_ori = extract_bv_ori(input_img)
    results_bf = crack_detection(input_img, use_bilateralFilter=True, use_gauss=True, use_erode3=False,
                                 use_cut200=True,
                                 check_shape=False)

    fig, axs = plt.subplots(1, 2, figsize=(18,9))
    axs[0].imshow(results_ori, cmap='gray')
    axs[1].imshow(results_bf, cmap='gray')
    # axs[2].imshow(results_bf, cmap='gray')
    plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.1, rect=None)
    plt.show()


    test_img_folder=False
    if test_img_folder:
        # pathFolder = r"D:\\data\\CHASEDB1\\train\\"
        pathFolder = "D:\\data\\inputs\\"
        filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder, x))]
        destinationFolder = r"D:\\data\\CHASEDB1\\outputs\\"
        if not os.path.exists(destinationFolder):
            os.mkdir(destinationFolder)
        for file_name in filesArray:
            file_name_no_extension = os.path.splitext(file_name)[0]
            input_img = cv2.imread(pathFolder + '/' + file_name)
            results_ori = extract_bv_ori(input_img)
            results_bf = crack_detection(input_img, use_bilateralFilter=True, use_gauss=True, use_erode3=False,
                                         use_cut200=True,
                                         check_shape=False)
            save_result = False
            if save_result == True:
                cv2.imwrite(destinationFolder + file_name_no_extension + "_bf.png", results_bf)

            fig, axs = plt.subplots(1, 2, figsize=(17, 9))
            axs[0].imshow(results_ori, cmap='gray')
            axs[1].imshow(results_bf, cmap='gray')
            # axs[2].imshow(results_bf, cmap='gray')
            plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.1, rect=None)
            plt.show()
