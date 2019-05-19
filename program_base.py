# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
from pytesseract import Output
import pandas as pd

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
image_size = h*w
mser = cv2.MSER_create()
mser.setMaxArea(int(image_size/2))
mser.setMinArea(10)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imwrite("gray.jpg", gray)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
thresh1 = cv2.adaptiveThreshold(tophat,255,1 ,1,9,2)
edged = cv2.Canny(gray, 10, 250)
_, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

regions, rects = mser.detectRegions(bw)
word_list = []
box_list  = []
hulls = []
# With the rects you can e.g. crop the letters
#https://stackoverflow.com/questions/51260994/error-performing-convexhull-on-mser-detected-regions
for (x, y, w, h) in rects:
    orig = image.copy()
    for p in regions:
        p = np.array(p)
        hulls.append( cv2.convexHull(p.reshape(-1, 1, 2)) ) 
        cv2.polylines(orig, hulls, 1, (0, 255, 0)) 
        cv2.rectangle(image, (x, y), (x+w, y+h), color=(255, 0, 255), thickness=1)
        box_list += [[x, y, w, h]]
        box_df = pd.DataFrame(box_list, columns=['left', 'top', 'width', 'height'])
        box_df.to_csv('result.csv')
d = pytesseract.image_to_data(image, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (xt, yt, wt, ht) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(orig, (xt, yt), (xt + wt, yt + ht), (0, 255, 0), 2)    
    word_list += [[left, top, width, height]]
    word_df = pd.DataFrame(word_list, columns=['left', 'top', 'width', 'height']) 
    word_df.to_csv('result.csv')
df = pd.read_csv('result.csv')
# http://qaru.site/questions/1572415/compare-values-in-two-columns-of-data-frame
rows = list(frame[frame['left'] != frame['x'].index)
rows2 = list(frame[frame['top'] != frame['y'].index)
print('Аномалии по x' rows)
print('Аномалии по y' rows2)
#return box_df
cv2.imwrite("tesser.jpg",orig)
cv2.imwrite("result_mine.jpg",image)
#cv2.imshow("Image", image)
#cv2.waitKey(0)
