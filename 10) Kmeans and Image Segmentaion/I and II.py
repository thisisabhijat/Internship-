import matplotlib.pyplot as plt
import cv2 

img = cv2.imread('sandreas.png')
print(img)

cv2.imshow('copy_sandreas', img)


cv2.waitKey(0)
cv2.destroyAllWindows()

