import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

screenshot_img = cv.imread('imgs/screenshot.png', cv.IMREAD_UNCHANGED)
wood_img = cv.imread('imgs/wood.jpg', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(screenshot_img, wood_img, cv.TM_CCOEFF_NORMED)


print(result)
threshold = .9
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))
print(locations)

if locations:
    print('found wood')

    wood_w = wood_img.shape[1]
    wood_h = wood_img.shape[0]
    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + wood_w, top_left[1] + wood_h)

        cv.rectangle(screenshot_img, top_left, bottom_right, line_color, line_type)
    cv.imshow('Matches', screenshot_img)
    cv.waitKey()
else:
    print('not found')
