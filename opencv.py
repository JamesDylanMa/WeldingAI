import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Lenna.png')

if image is None:
    print("Image load failed!")
    sys.exit()

# image 흑백, 컬러

image1 = cv2.imread('Lenna.png', cv2.IMREAD_UNCHANGED)  # 원본
image2 = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)  # 흑백
image3 = cv2.imread('Lenna.png', cv2.IMREAD_COLOR)      # 컬러

image4 = cv2.imread('Lenna.png', 2)

# # RGB타입으로 변환
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# # Resize
# # image5 = cv2.resize(image,(1280,1080))
# image5_1 = cv2.resize(image,(1280,1080), interpolation=cv2.INTER_NEAREST)
# image5_2 = cv2.resize(image,(1280,1080), interpolation=cv2.INTER_LINEAR)
# image5_3 = cv2.resize(image,(1280,1080), interpolation=cv2.INTER_CUBIC)
# image5_4 = cv2.resize(image,(1280,1080), interpolation=cv2.INTER_LANCZOS4)
# image5_5 = cv2.resize(image,(1280,1080), interpolation=cv2.INTER_LINEAR_EXACT)

# # Pre-processing
# image6 = cv2.equalizeHist(image2) # only for grayscale
# image7 = cv2.filter2D(image,-1,1.5) #filter
# image7_2 = cv2.GaussianBlur(image,(0,0), 2) #GaussianBlur


# # Image data Augmentation
# image_cropped = image[:,:256]
# image_cropped2 = image[:256,:]
# image_blur = cv2.blur(image,(10,10)) #blur image

# height, width, channel = image.shape
# matrix = cv2.getRotationMatrix2D((width/2,height/2), 45, 0.5)  # (center, angle, scale) // angle + -> 반시계
# image_rotate = cv2.warpAffine(image, matrix,(width, height))

# # Image_Line & Polygon 그리기
# cv2.line(image,(0,0),(256,256), (255,0,0)) # img, 시작점좌표(x,y), 종료점좌표(x,y),color(255,0,0) = blue 
# cv2.rectangle(image,(10,10),(100,100),(255,255,0),-1) # 사각형 그리기

# points1 = np.array([[110,110],[270,110],[300,330],[170,170],[150,250]],np.int32)
# image_poly = cv2.polylines(image,[points1],True,(255,0,0), 2) # Polygons

# # 텍스트 넣기
# cv2.putText(image, 'Lenna', (360, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
# # 텍스트를 넣을 이미지, 텍스트 내용, 텍스트 시작 좌측하단좌표, 글자체, 글자크기, 글자색, 글자두께, cv2.LINE_AA(좀 더 예쁘게 해주기 위해)
# cv2.putText(image, 'Lenna', (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(image, 'Lenna', (400, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(image, 'Lenna', (120, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 

cv2.imshow('test', image)
cv2.imshow('original', image1)
cv2.imshow('gray', image2)
cv2.imshow('color', image3)
cv2.imshow('image4',image4)
# cv2.imshow('compare',image5)
# cv2.imshow('EqulHist',image6) # gray image only
# cv2.imshow('Sharp',image7) # filter
# cv2.imshow('cropped image', image_cropped)
# cv2.imshow('cropped image_2',image_cropped2)
# cv2.imshow('Blur image', image_blur)
# cv2.imshow('Rotated Image', image_rotate)
# cv2.imshow('Polygons',image_poly)
# cv2.imshow('G_B',image7_2)


# cv2.imwrite('Lenna_gray.png', image4)   #이미지 저장

# capture = cv2.VideoCapture(0)  # 0: 내장 카메라, 1: 외장 카메라
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #640
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #480

# while cv2.waitKey(33) < 0:
#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)

# capture.release()
# cv2.destroyAllWindows()


cv2.waitKey(0)    # 0: 무한반복, delay time


# while True:
#     if cv2.waitKey() == 27:  # esc(27), enter(13), tap(9)   ,    ord('q')
#         break


cv2.destroyAllWindows()
