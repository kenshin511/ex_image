# -*- coding: utf-8 -*-
"""
ex_canny
Created on Mon Jan 16 13:14:17 2017
2017/01/16 ~ 2017/01/18
@author: cho
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==============================Def============================================
def canny_threshold(mag, r, c, highTr, lowTr, cand_list, pEdge):
    STRONG_EDGE = 255
    WEAK_EDGE = 128
    
    cand_list.append([r, c])
    if mag[r][c] > highTr:
        pEdge[r][c] = STRONG_EDGE
    elif mag[r][c] > lowTr:
        pEdge[r][c] = WEAK_EDGE
    
# -----------------------------------------------------------------------------
def non_maximum(ang, mag, highThresh, lowThresh, cand_list, pEdge):
    hgt, wid = mag.shape[:]
    for r in range(1, hgt-1):
        for c in range(1, wid-1):
            # 0도 방향
            if ang[r][c] == 0:
                if mag[r][c] > mag[r][c-1] and mag[r][c] > mag[r][c+1]: 
                    canny_threshold(mag, r, c, highThresh, lowThresh, 
                                    cand_list, pEdge)
                        
            # 45도 방향 
            elif ang[r][c] == 45:
                if mag[r][c] > mag[r-1][c-1] and mag[r][c] > mag[r+1][c+1]:
                    canny_threshold(mag, r, c, highThresh, lowThresh, 
                                    cand_list, pEdge)
                        
            # 90도 방향 
            elif ang[r][c] == 90:
                if mag[r][c] > mag[r-1][c] and mag[r][c] > mag[r+1][c]:
                    canny_threshold(mag, r, c, highThresh, lowThresh, 
                                    cand_list, pEdge)
                    
            # 135 방향 
            elif ang[r][c] == 135:
                if mag[r][c] > mag[r+1][c-1] and mag[r][c] > mag[r-1][c+1]:
                    canny_threshold(mag, r, c, highThresh, lowThresh, 
                                    cand_list, pEdge)               
    
# -----------------------------------------------------------------------------
def normalize_angle(ang):
    heigt, width = ang.shape[:]
    for r in range(heigt):
        for c in range(width):
            theta = ang[r][c]
            if (theta > -22.5 and theta < 22.5) or\
            theta > 157.5 or theta < -157.5:
                ang[r][c] = 0

            elif (theta >= 22.5 and theta < 67.5) or\
            (theta >= -157.5 and theta < -112.5):
                ang[r][c] = 45

            elif (theta >= 67.5 and theta <= 112.5) or\
            (theta >= - 112.5 and theta <= -67.5):
                ang[r][c] = 90

            else:
                ang[r][c] = 135
    ang = ang.astype(int)
    
# -----------------------------------------------------------------------------
def ang_neighbor(ang, r, c):
    if ang[r][c] == 0:
        r1, r2 = r, r
        c1, c2 = c-1, c+1
    elif ang[r][c] == 45:
        r1, r2 = r+1, r-1
        c1, c2 = c-1, c+1        
    elif ang[r][c] == 90:        
        r1, r2 = r-1, r+1
        c1, c2 = c, c
    elif ang[r][c] == 135:
        r1, r2 = r-1, r+1
        c1, c2 = c-1, c+1        
        
    out = [[r1, c1], [r2, c2]]
    return out   

# ============================================================================    
# ================================Main========================================    
path = 'D:\OpenCV\sample\\'
img = cv2.imread(path + 'star2.jpg')
f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Magnitude
sobelx = cv2.Sobel(f, cv2.CV_16S, 1, 0, ksize = 5)
sobely = cv2.Sobel(f, cv2.CV_16S, 0, 1, ksize = 5)
mag = np.hypot(sobelx, sobely)    
mag = mag/mag.max() * 255
# Angle    
ang = np.arctan2(sobely, sobelx) * 180/np.pi

# norm_angle
normalize_angle(ang)

# Non maximum
cand_list = list()
threshHi, threshLow = 150, 50
pEdge = np.zeros_like(mag, np.uint8)
                
non_maximum(ang, mag, threshHi, threshLow, cand_list, pEdge)

# Hysteris 
threshHi = 200
threshLow = 100
out = np.zeros_like(f)
for r in range(pEdge.shape[0]):
    for c in range(pEdge.shape[1]):
        if pEdge[r][c]:
            if pEdge[r][c] == 255:
                out[r][c] = 255
            elif pEdge[r][c] == 128: # low threshold
                # 0도 방향 
                if ang[r][c] == 90: 
                    if pEdge[r][c-1] == 255 or pEdge[r][c+1] == 255: 
                        out[r][c] = 255  
              
                # 45도 방향 
                elif ang[r][c] == 45:                        
                    if pEdge[r+1][c-1] == 255 or pEdge[r-1][c+1] == 255: 
                        out[r][c] = 255

                # 90도 방향
                elif ang[r][c] == 0:  
                    if pEdge[r-1][c] == 255 or pEdge[r+1][c] == 255:
                        out[r][c] = 255

                # 135도 방향 
                elif ang[r][c] == 135:
                    if pEdge[r-1][c-1] == 255 or pEdge[r+1][c+1] == 255: 
                        out[r][c] = 255

out[out == 128] = 0
                    
# Figure
plt.imshow(out)
plt.show()

plt.imshow(f)
plt.show()

cv2.imshow('o', out)
cv2.waitKey()
            
