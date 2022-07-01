import numpy as np
import cv2

"""
    Toán tử giãn nở ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau giãn nở 
"""
def dilate(img, kernel):
    # Tính thông số cơ bản
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    img_shape = img.shape
    dilated_img = np.zeros((img_shape[0], img_shape[1]))

    # Mở rộng ảnh gốc về 4 phía để có thể quét
    x_append = np.zeros((img.shape[0], kernel_center[1]))
    img = np.append(x_append, img, axis=1)
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel_center[0], img.shape[1]))
    img = np.append(y_append, img, axis=0)
    img = np.append(img, y_append, axis=0)
    
    # Xét từng điểm của ảnh gốc
    # Cộng điểm đó và các điểm xung quanh thêm b rồi tìm giá trị lớn nhất
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            # Tăng giá trị tại i và các điểm lân cận
            tmp_img = np.zeros((kernel.shape[0], kernel.shape[1]))
            tmp_img = img[i:i + kernel.shape[0], j:j + kernel.shape[1]] + kernel
            # Chuẩn hoá lại tmp_img nếu có giá trị nào vượt quá 255
            for k in range(tmp_img.shape[0]):
                for l in range(tmp_img.shape[1]):
                    if tmp_img[k, l] > 255:
                        tmp_img[k, l] = 255
            # Tìm max và gán vào ảnh kết quả
            dilated_img[i, j] = np.amax(tmp_img)
            
    return dilated_img

"""
    Toán tử co ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau co 
"""
def erode(img, kernel):
    # Tính thông số cơ bản
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    img_shape = img.shape
    eroded_img = np.zeros((img_shape[0], img_shape[1]))

    # Mở rộng ảnh gốc về 4 phía để có thể quét
    x_append = np.zeros((img.shape[0], kernel_center[1]))
    img = np.append(x_append, img, axis=1)
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel_center[0], img.shape[1]))
    img = np.append(y_append, img, axis=0)
    img = np.append(img, y_append, axis=0)
    
    # Xét từng điểm của ảnh gốc
    # Cộng điểm đó và các điểm xung quanh thêm b rồi tìm giá trị lớn nhất
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            # Tăng giá trị tại i và các điểm lân cận
            tmp_img = np.zeros((kernel.shape[0], kernel.shape[1]))
            tmp_img = img[i:i + kernel.shape[0], j:j + kernel.shape[1]] - kernel
            # Chuẩn hoá lại tmp_img nếu có giá trị nào nhỏ hơn 0
            for k in range(tmp_img.shape[0]):
                for l in range(tmp_img.shape[1]):
                    if tmp_img[k, l] < 0:
                        tmp_img[k, l] = 0
            # Tìm max và gán vào ảnh kết quả
            eroded_img[i, j] = np.amin(tmp_img)
            
    return eroded_img

"""
    Toán tử giãn mở ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau mở 
"""
def opening(img, kernel):
    # Áp dụng co ảnh xám
    tmp_img = erode(img, kernel)

    # Áp dụng giãn nở ảnh xám
    opening_img = dilate(tmp_img, kernel)

    return opening_img

"""
    Toán tử đóng ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau đóng 
"""
def closing(img, kernel):
    # Áp dụng giãn nở ảnh xám
    tmp_img = dilate(img, kernel)

    # Áp dụng co ảnh xám
    closing_img = erode(tmp_img, kernel)

    return closing_img

"""
    Toán tử hồi phục ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau hồi phục
"""
def reconstruction(img, kernel):

    return None

"""
    Toán tử Gradient ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau thực hiện gradient
"""
def gradient(img, kernel):
    # Áp dụng giãn nở ảnh xám
    dilated_img = dilate(img, kernel)

    # Áp dụng co ảnh xám
    eroded_img = erode(img, kernel)

    # Thực hiện trừ ảnh giãn nở cho ảnh co
    gradient_img = dilated_img - eroded_img
    return gradient_img

"""
    Toán tử Top-Hat ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau thực hiện Top-hat
"""
def topHat(img, kernel):
    # Áp dụng mở ảnh xám
    opening_img = opening(img, kernel)

    # Thực hiện trừ ảnh gốc cho ảnh mở
    topHat_img = img - opening_img
    return topHat_img

"""
    Toán tử Black-Hat ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau thực hiện Black-hat
"""
def blackHat(img, kernel):
    # Áp dụng đóng ảnh xám
    closing_img = closing(img, kernel)

    # Thực hiện trừ ảnh gốc cho ảnh đóng
    blackHat_img = img - closing_img
    return blackHat_img

"""
    Toán tử Textual Segmentation ảnh xám
    Input: ảnh xám, kernel
    Output: ảnh xám sau thực hiện Textual Segmentation
"""
def textualSegmentation(img, kernel):
    # Áp dụng đóng ảnh xám
    textualSegmentation_img = closing(img, kernel)

    # Áp dụng mở ảnh xám
    textualSegmentation_img = opening(textualSegmentation_img, kernel)

    return textualSegmentation_img