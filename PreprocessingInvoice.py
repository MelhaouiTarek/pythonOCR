# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:42:59 2022

@author: Xune
"""

from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re
from PIL import Image
import numpy as np
import pkg_resources


orig = cv2.imread("C:\\Users\\Xune\\Pictures\\test3.jpg")
image=orig.copy()
image = imutils.resize(image,width=500)
ratio = orig.shape[1] / float(image.shape[1])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged= cv2.Canny(blurred, 75, 200)


cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)

r4 = None
r4Found = False



for c in cnts:
    peri=cv2.arcLength(c, True)
    approx=cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        r4  = approx
        break
        

if r4 is None:
    print("bruh")


output2 = image.copy()
cv2.drawContours(output2, [r4], -1, (0,255,0),2)
receipt=four_point_transform(orig, r4.reshape(4,2)*ratio)

cv2.imwrite("temp.jpg",  imutils.resize(receipt,width=500))


kernel = np.ones((5,5),np.float32)/25



image_to_ocr = cv2.resize(receipt,None,1.5,1.5,cv2.INTERSECT_FULL)

preprocessed_img = cv2.cvtColor(image_to_ocr, cv2.COLOR_BGR2GRAY)


preprocessed_img = cv2.erode(preprocessed_img,(5,5))



preprocessed_img = cv2.GaussianBlur(preprocessed_img, (3,3), 0)

preprocessed_img = cv2.adaptiveThreshold(preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
preprocessed_img= cv2.bitwise_not(preprocessed_img)
preprocessed_img = cv2.dilate(preprocessed_img,(5,5), iterations=1)



cv2.imwrite("temp2.jpg", preprocessed_img)

pil = Image.open("temp2.jpg")
text = pytesseract.image_to_string(pil)

print(text)