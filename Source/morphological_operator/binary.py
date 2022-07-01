import numpy as np
import cv2


def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 1

    return eroded_img[:img_shape[0], :img_shape[1]]

'''
TODO: implement morphological operators
'''

"""
    Toán tử giãn nở nhị phân
    Input: ảnh nhị phân, kernel
    Output: ảnh nhị phân sau giãn nở 
"""
def dilate(img, kernel):
    #Tính thông số cơ bản
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    dilated_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    # Xét từng điểm của ảnh gốc
    # Nếu điểm đó là điểm trắng thì mở rộng theo kernel
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if (img[i, j])==255:
                dilated_img[i:i + kernel.shape[0], j:j + kernel.shape[1]] = kernel

    return dilated_img[kernel_center[0]:(kernel_center[0] + img_shape[0]), kernel_center[1]:(kernel_center[1] + img_shape[1])]

"""
    Toán tử mở nhị phân
    Input: ảnh nhị phân, kernel
    Output: ảnh nhị phân sau mở
"""
def opening(img, kernel):
    # Áp dụng co nhị phân
    tmp_img = erode(img, kernel)

    # Đưa ảnh nhị phân sau khi co về mức 0,255
    for i in range(tmp_img.shape[0]):
        for j in range(tmp_img.shape[1]):
            if tmp_img[i, j] == 1:
                tmp_img[i, j] = 255

    # Áp dụng giãn nở nhị phân
    opening_img = dilate(tmp_img, kernel)

    return opening_img

"""
    Toán tử đóng nhị phân
    Input: ảnh nhị phân, kernel
    Output: ảnh nhị phân sau đóng 
"""
def closing(img, kernel):
    # Áp dụng giãn nở nhị phân
    tmp_img = dilate(img, kernel)

    # Đưa ảnh nhị phân sau khi co về mức 0,255
    for i in range(tmp_img.shape[0]):
        for j in range(tmp_img.shape[1]):
            if tmp_img[i, j] == 1:
                tmp_img[i, j] = 255

    # Áp dụng co nhị phân
    closing_img = erode(tmp_img, kernel)

    return closing_img

"""
    Toán tử Hit-or-Miss
    Input: ảnh nhị phân, kernel
    Output: ảnh nhị phân sau hit-or-miss
"""
def hitOrMiss(img, kernel):
    #Tính thông số cơ bản
    hit_or_miss_img = np.zeros((img.shape[0], img.shape[1]))
    img_shape = img.shape

    # Tìm phần bù của ảnh
    compensation_img = np.zeros((img_shape[0], img_shape[1]))
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i, j] == 0:
                compensation_img[i ,j] = 255
    
    # Tạo phần bù của kernel (phải thêm các hàng 1 ở rìa của bốn phía)
    compensation_kernel = np.zeros((kernel.shape[0], kernel.shape[1]))
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if kernel[i, j] == 0:
                compensation_kernel[i, j] = 1
    x_append = np.ones((compensation_kernel.shape[0], 1))
    compensation_kernel = np.append(x_append, compensation_kernel, axis=1)
    compensation_kernel = np.append(compensation_kernel, x_append, axis=1)
    y_append = np.ones((1, compensation_kernel.shape[1]))
    compensation_kernel = np.append(y_append, compensation_kernel, axis=0)
    compensation_kernel = np.append(compensation_kernel, y_append, axis=0)
    
    # Áp dụng co nhị phân cho ảnh gốc và ảnh phần bù
    eroded_img = erode(img, kernel)
    eroded_compensation_img = erode(compensation_img, compensation_kernel)

    # Tìm phần giao của hai ảnh sau khi co nhị phân
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if eroded_img[i, j] == 1 and eroded_compensation_img[i + 1, j + 1] == 0:
                hit_or_miss_img[i, j] = 1

    return hit_or_miss_img

"""
    Toán tử Thinning
    Input: ảnh nhị phân, kernel
    Output: ảnh nhị phân sau thinning
"""
def thinning(img, kernel):
    # Tính hit-or-miss của ảnh
    hit_or_miss_img = hitOrMiss(img, kernel)

    # Trừ ảnh gốc cho ảnh sau hit-or-miss
    thinning_img = img // 255 - hit_or_miss_img

    return thinning_img

"""
    Toán tử Boundary Extraction
    Input: ảnh nhị phân, kernel
    Output: ảnh nhị phân sau hit-or-miss
"""
def boundaryExtraction(img, kernel):
    # Tính erode của ảnh
    eroded_img = erode(img, kernel)

    # Trừ ảnh gốc cho ảnh sau erode
    boundaryExtraction_img = img - eroded_img

    return boundaryExtraction_img