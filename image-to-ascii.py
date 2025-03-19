
threshold = 0.2
import sys
if len(sys.argv) > 1:
    try:
        threshold = float(sys.argv[1])
    except:
        print('unparsable', file=sys.stderr)
        sys.exit(1)

import string

print(string.printable)

import cv2

# image must not be transparent or OpenCV stacks across runs
img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# show image
cv2.imshow('result', img)

imts = {}
from itertools import chain
for xc in chain(string.ascii_letters, string.digits, string.punctuation,' '):
    # Reading an image in default mode
    imt = cv2.imread('letter.png', cv2.IMREAD_GRAYSCALE)

    # Window name in which image is displayed
    window_name = 'letter-'+xc

    # text
    text = xc

    # font
    font = cv2.FONT_HERSHEY_COMPLEX

    # org
    org = (1, 6)

    # fontScale
    fontScale = 0.25
 
    # color in BGR
    color = (0, 0, 0)

    # Line thick2ness in px
    thickness = 1
 
    # Using cv2.putText() method
    image = cv2.putText(imt, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)

    # Displaying the image
    #cv2.imshow(window_name, image)
    imts[xc] = image

# draw ascii
img_h, img_w = img.shape[:2]
imt_h, imt_w = imt.shape[:2]

import numpy as np
import random
for y in range(img_h//imt_h):
    for x in range(img_w//imt_w):
        imc = img[y*imt_h:(y+1)*imt_h,x*imt_w:(x+1)*imt_w]
        #cv2.imshow('{}-{}'.format(x,y),imc)
        m_r = (' ', 0)
        cs = list(imts.keys())
        random.shuffle(cs)
        for c, i in [(c, imts[c]) for c in cs]:
            r = cv2.matchTemplate(imc,i,cv2.TM_CCOEFF_NORMED)
            mm_r = max(chain.from_iterable(r))
            loc = np.where(r > threshold)
            if len(loc[0]):
                print(c,end='')
                m_r = (c, mm_r)
                break
        else:
            print('#',end='')
    print()

cv2.waitKey(5000) # wait milliseconds

cv2.destroyAllWindows()

