import numpy as np
import matplotlib.pyplot as plt

image = plt.imread("C:/example/bbb.jpg") #이미지 불러오기

def calc_hsv(bgr):
    B, G, R = float(bgr[0]), float(bgr[1]), float(bgr[2])       # B, G, R 값 저장

    V = max(R, G, B) # R,G,B중 최댓값을 V에 저장
    if V != 0 :
        S = V - min(R,G,B)/V #수식을 통해 S 계산
    else:
        S = 0

    if S==0: #S가 0일 경우 H 계산식 분모가 0이 되므로 H도 0으로 설정
        H = 0
    elif V== R:
        H = (G-B)*60 / S
    elif V == G:
        H = (G-B)*60 / S + 120
    elif V == B:
        H = (G-B)*60 /S + 240
    return (H, S, V)


def bgr2hsv(image):
    copy = np.copy(image) # 변환 이미지 저장 공간 생성
    for i in range(0, 800):
        for j in range(0, 600):
            hsv = calc_hsv(image[i,j])
            copy[i,j] = hsv # hsv 변환된 픽셀 값 저장
    return (np.array(copy)).astype('uint8')

def calc_histo(image, histX, histY): #히스토그램 계산 함수
    hist = np.zeros((histX//10+1, histY//10+1), np.uint8) #빈도 수 저장 공간 (256,256) 생성
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            hist[image[i,j,0]//10,image[i,j,1]//10] += 1 # H, S 빈도 수 체크
    return hist

hsvImg = bgr2hsv(image) #bgr->hsv 변환
hist = calc_histo(hsvImg, 256, 256) #가로 S, 세로 H의 빈도 수 계산

plt.pcolor(hist)
plt.show()
plt.title('hsvImg');plt.imshow(hsvImg)
plt.show()