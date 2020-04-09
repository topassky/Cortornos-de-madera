# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import div
import cv2
import numpy as np
import imutils
import cuencas 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image = imutils.resize(image, width = 600)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))
# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = div.centroid_histogram(clt)
bar , porcentaje, color = div.plot_colors(hist, clt.cluster_centers_)
color = color[list(porcentaje).index(max(porcentaje))]
color  = np.asarray(color)
colorHSV = np.zeros(len(color))

colorHSV[0], colorHSV[1], colorHSV[2]=div.rgb_to_hsv(color[0], color[1], color[2])
print(colorHSV)
H20 = colorHSV[1]*0.80
S20 = colorHSV[1]*0.80
V20 = colorHSV[2]*0.80
lower = np.zeros(len(color))
upper = np.zeros(len(color))
lower[0], lower[1], lower[2] = colorHSV[0]-H20, colorHSV[1]-H20, colorHSV[2]-V20
upper[0], upper[1], upper[2] = colorHSV[0]+H20, 255, 255

lower = np.array([lower[0], lower[1], lower[2]])
upper = np.array([upper[0], upper[1], upper[2]])

img = cv2.imread(args["image"])
img = imutils.resize(img, width = 600)
converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)
# apply a series of erosions and dilations to the mask
# using an elliptical kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(skinMask, kernel, iterations = 2)
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
# blur the mask to help remove noise, then apply the
# mask to the frame
#skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(img, img, mask = skinMask)
# show the skin in the image along with the mask
image = cv2.imread(args["image"])
image = imutils.resize(image, width = 600)
cuencas.encontrar_contornos(skin, image)
# show our color bart
#plt.figure()
#plt.axis("off")
#plt.imshow(bar)
#plt.show()
#cv2.imshow("images",skin)
#cv2.waitKey(0)

