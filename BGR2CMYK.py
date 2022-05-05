import numpy as np
import matplotlib.pyplot as plt

image = plt.imread("C:/example/bbb.jpg")

def calc_YCbCr(bgr): # BGR -> YCbCr 변환 함수
    B, G, R = float(bgr[0]), float(bgr[1]), float(bgr[2]) #B,G,R 값 저장
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B-Y) * 0.564 + 128
    Cr = (R-Y) * 0.713 + 128

    R = Y + 1.408*(Cr-128)
    G = Y - 0.714*(Cr-128) - 0.344*(Cb - 128)
    B = Y + 1.773*(Cb-128) # 공식을 통해 변환
    return ([B, G, R])


def bgr2YCbCr(image):
    copy = np.copy(image) #변환 이미지 저장 공간 생성
    for i in range(0, 800):
        for j in range(0, 600):
            bgr = calc_YCbCr(image[i,j]) 
            copy[i,j] = bgr # 변환한 값을 각 픽셀에 저장
    return (np.array(copy)).astype('uint8')

YCbCr_img = bgr2YCbCr(image)

print(image[0,0,:])
print(YCbCr_img[0,0,:])
plt.title('YCbCr_img');plt.imshow(YCbCr_img)
plt.show()