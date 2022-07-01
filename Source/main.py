import sys
import getopt
import cv2
import numpy as np
from morphological_operator import binary
from morphological_operator import grayscale


def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)

    kernel_gray = np.ones((3, 3), np.uint8)
    kernel_gray = kernel_gray * 10

    img_out = None

    '''
    TODO: implement morphological operators
    '''
    if mor_op == 'dilate':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)

        img_dilation_gray = cv2.dilate(img_gray, kernel_gray)
        cv2.imshow('OpenCV dilation image grayscale', img_dilation_gray)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)

        img_dilation_gray_manual = grayscale.dilate(img_gray, kernel_gray)
        cv2.imshow('manual dilation grayscale image', img_dilation_gray_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_gray_manual
    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)

        img_erosion_gray = cv2.erode(img_gray, kernel_gray)
        cv2.imshow('OpenCV erosion image grayscale', img_erosion_gray)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)

        img_erosion_gray_manual = grayscale.erode(img_gray, kernel_gray)
        cv2.imshow('manual erosion image grayscale', img_erosion_gray_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_gray_manual
    elif mor_op == "opening":
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)

        img_opening_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel_gray)
        cv2.imshow('OpenCV opening image grayscale', img_opening_gray)
        cv2.waitKey(wait_key_time)

        img_opening_manual = binary.opening(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)

        img_opening_gray_manual = grayscale.opening(img_gray, kernel_gray)
        cv2.imshow('manual opening image grayscale', img_opening_gray_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_gray_manual
    elif mor_op == "closing":
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closing image', img_closing)

        img_closing_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel_gray)
        cv2.imshow('OpenCV closing image grayscale', img_closing_gray)
        cv2.waitKey(wait_key_time)

        img_closing_manual = binary.closing(img, kernel)
        cv2.imshow('manual closing image', img_closing_manual)

        img_closing_gray_manual = grayscale.closing(img_gray, kernel_gray)
        cv2.imshow('manual closing image grayscale', img_closing_gray_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_gray_manual
    elif mor_op == "hit-or-miss":
        img_hit_or_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS,kernel)
        cv2.imshow('OpenCV hit-or-miss image', img_hit_or_miss)
        cv2.waitKey(wait_key_time)

        img_hit_or_miss_manual = binary.hitOrMiss(img, kernel)
        cv2.imshow('manual hit-or-miss image', img_hit_or_miss_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hit_or_miss_manual
    elif mor_op == "thinning":
        img_thinning = cv2.ximgproc.thinning(img)
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)

        img_thinning_manual = binary.thinning(img, kernel)
        cv2.imshow('manual thinning image', img_thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual
    elif mor_op == "boundary-extraction":
        img_boundary_extraction_manual = binary.boundaryExtraction(img, kernel)
        cv2.imshow('manual boundary extraction image', img_boundary_extraction_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_boundary_extraction_manual
    elif mor_op == "mor-gradient":
        img_gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel_gray)
        cv2.imshow('OpenCV gradient image', img_gradient)
        cv2.waitKey(wait_key_time)

        img_gradient_manual = grayscale.gradient(img_gray, kernel_gray)
        cv2.imshow('manual gradient image', img_gradient_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gradient_manual
    elif mor_op == "top-hat":
        img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel_gray)
        cv2.imshow('OpenCV top-hat image grayscale', img_tophat)
        cv2.waitKey(wait_key_time)

        img_tophat_manual = grayscale.topHat(img_gray, kernel_gray)
        cv2.imshow('manual top-hat image grayscale', img_tophat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_tophat_manual
    elif mor_op == "black-hat":
        img_blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT,kernel_gray)
        cv2.imshow('OpenCV black-hat image grayscale', img_blackhat)
        cv2.waitKey(wait_key_time)

        img_blackhat_manual = grayscale.blackHat(img_gray, kernel_gray)
        cv2.imshow('manual black-hat image grayscale', img_blackhat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_blackhat_manual
    elif mor_op == "textual-segmentation":
        img_textual_segmentation_manual = grayscale.textualSegmentation(img_gray, kernel_gray)
        cv2.imshow('manual img_tophat image grayscale', img_textual_segmentation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_textual_segmentation_manual
    

    if img_out is not None:
        cv2.imwrite(out_file, img_out)


def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])
