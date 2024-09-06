import cv2 as cv
import numpy as np
from numpy import linalg as LA
import time

def cropImageBGR(img):
    height = img.shape[0]
    h = int(np.floor(height/3.0))
    return img[:h], img[h:2*h], img[2*h:3*h]

def computeCost(img1, img2, dx,dy):
    '''
    shift img1 by dx, dy and compute the L2 norm of the image difference
    '''
    #assert img1.shape==img2.shape
    h, w = img1.shape
    # crop the image border, this make the process much faster and reduce the border's
    h_offset = int(h/4)
    w_offset = int(w/4)
    # compare overlap region of image1 and 2
    if dx<0:
        img1_x1 = -dx
        img1_x2 = w-1
        img2_x1 = 0
        img2_x2 = w+dx-1
    else:
        img1_x1 = 0
        img1_x2 = w-dx-1
        img2_x1 = dx
        img2_x2 = w-1

    if dy<0:
        img1_y1 = -dy
        img1_y2 = h-1
        img2_y1 = 0
        img2_y2 = h+dy-1
    else:
        img1_y1 = 0
        img1_y2 = h-dy-1
        img2_y1 = dy
        img2_y2 = h-1
     
    roi1 = img1[img1_y1+h_offset:img1_y2-h_offset,img1_x1+w_offset:img1_x2-w_offset]
    roi2 = img2[img2_y1+h_offset:img2_y2-h_offset,img2_x1+w_offset:img2_x2-w_offset]

    # compute L2 norm
    diff = roi1-roi2
    return LA.norm(diff) # same as np.sqrt(np.sum(diff**2)), but faster

def generateImagePyramids(img, max_scale = 4):
    '''
    Generate image pyramids by halving the images
    '''
    h, w = img.shape
    max_scale = int(min(np.floor(np.log2(h)),np.floor(np.log2(w)),max_scale)) # check max scale
    pyramid = []
    for scale in range(max_scale):
        factor = 1/(2**scale)
        h_ = int(h*factor)
        w_ = int(w*factor)
        # prevent the image become too small (too small can make the result worse)
        if(h_ < 180 or w_<180):
            break
        pyramid.append(cv.resize(img, (w_, h_)).astype(np.float32))
    return pyramid

def findAlignmentSingleScale(img1, img2, dx=0, dy=0, search_window = (-15,15)):
    assert img1.shape==img2.shape
    im1 = img1.astype(np.float32)
    im2 = img2.astype(np.float32)
    # pre-posess imges
    # normalize image by mapping image pixel value to 0-1
    a1 = np.min(im1)
    b1 = np.max(im1)
    a2 = np.min(im2)
    b2 = np.max(im2)
    im1 = (im1-a1)/(b1-a1)
    im2 = (im2-a2)/(b2-a2)
    # use 0.5 as image 2 BW threshold, compute the image 1 threshold
    th1 = 0.5*np.mean(im1)/np.mean(im2)
    _,im1 = cv.threshold(im1, th1, 1, cv.THRESH_BINARY )
    _,im2 = cv.threshold(im2, 0.5, 1, cv.THRESH_BINARY )

    # compute serach window aroung the current dx, dy
    range_dx = range(dx+search_window[0],dx+search_window[1]+1)
    range_dy = range(dy+search_window[0],dy+search_window[1]+1)
    best_dx = dx
    best_dy = dy
    best_cost = np.Inf

    #find du, dv that has the minimu cost
    for dx in range_dx:
        for dy in range_dy:
            cost  =computeCost(im1,im2,dx,dy) # use Eucilid distance of image different
            if cost<best_cost:
                best_cost = cost
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy

def findAlignmentMultiScale(pyramid1, pyramid2, scale, dx=0, dy=0, search_window = (-15,15)):
    # recursion guard
    if scale<0:
        return dx,dy
    
    # input dx, dy are estimte from last scale, map them to current scale 
    dx*=2
    dy*=2
    # retrive image from pyramid and convert to float
    img1 = pyramid1[scale]
    img2 = pyramid2[scale]
    
    best_dx, best_dy = findAlignmentSingleScale(img1,img2,dx,dy,search_window)

    return findAlignmentMultiScale(pyramid1, pyramid2, scale-1, best_dx, best_dy) # recursive call

def alignImagesSingleScale(img1,img2):
    best_dx, best_dy = findAlignmentSingleScale(img1, img2, 0,0,(-15,15))
    h,w = img2.shape
    print(f'Displacement vector: ({best_dx,best_dy})')
    return cv.warpAffine(img1, np.array([[1,0,best_dx],[0,1,best_dy]]).astype(np.float32), (w,h))

def alignImagesMultiScale(pyramid1,pyramid2):
    assert len(pyramid1)==len(pyramid2)
    scale = len(pyramid1)-1
    best_dx, best_dy = findAlignmentMultiScale(pyramid1, pyramid2, scale, 0,0,(-15,15))
    h,w = pyramid2[0].shape
    print(f'Displacement vector: ({best_dx,best_dy})')
    return cv.warpAffine(pyramid1[0], np.array([[1,0,best_dx],[0,1,best_dy]]).astype(np.float32), (w,h))

def test_single_scale():
    filepath_ss=[]
    # small size, single scale
    filepath_ss.append("images/cathedral.jpg")
    filepath_ss.append("images/monastery.jpg")
    filepath_ss.append("images/tobolsk.jpg")
    for filepath in filepath_ss:
        # single scale
        t0 = time.time()
        filename = filepath.split('/')[-1].split('.')[0]
        print(filename)
        # load image
        image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        # crop image
        b,g,r = cropImageBGR(image)
        # align image, single scale
        ag = alignImagesSingleScale(g,b)
        ar = alignImagesSingleScale(r,b)
        combine = np.dstack((b,ag,ar))
        cv.imwrite(f"./images_colorized/{filename}_c.jpg", combine)
        t1 = time.time()
        print(f'Total elapsed time: {t1-t0:.3}s\n')

def test_multi_scale():
    # large size, multi-scale
    filepath_ms=[]
    filepath_ms.append("images/lady.tif")
    filepath_ms.append("images/train.tif")
    filepath_ms.append("images/church.tif")
    filepath_ms.append("images/emir.tif")
    filepath_ms.append("images/harvesters.tif")
    filepath_ms.append("images/icon.tif")
    filepath_ms.append("images/melons.tif")
    filepath_ms.append("images/onion_church.tif")
    filepath_ms.append("images/sculpture.tif")
    filepath_ms.append("images/self_portrait.tif")
    filepath_ms.append("images/three_generations.tif")

    for filepath in filepath_ms:
        t0 = time.time()
        filename = filepath.split('/')[-1].split('.')[0]
        print(filename)

        # load image
        image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        b,g,r = cropImageBGR(image)
        # create pyramid
        b_pyd = generateImagePyramids(b,4)
        g_pyd = generateImagePyramids(g,4)
        r_pyd = generateImagePyramids(r,4)
        # align image, multi scale
        ag = alignImagesMultiScale(g_pyd,b_pyd)
        ar = alignImagesMultiScale(r_pyd,b_pyd)

        # ouput colorized image
        combine = np.dstack((b,ag,ar))
        cv.imwrite(f"./images_colorized/{filename}_c.jpg", combine)
        t1 = time.time()
        print(f'Total elapsed time: {t1-t0:.3}s\n')

if __name__ == "__main__":
    test_single_scale()
    test_multi_scale()