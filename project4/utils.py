import numpy as np
import cv2 as cv

######################### Image Multiband blending ##################################################

def createGaussianStack(im, lv, sz):
    '''
    create a gassian stack with the given level and the size of the gaussian filter
    '''
    img = np.copy(im)
    stackG=[img]
    for i in range(lv-1): # for lv layer, only convole lv-1 times
        img =  cv.GaussianBlur(img, sz, 0 ) # all three channel if RGB
        stackG.append(img)
    return stackG

def createLaplacianStack(im, lv, sz):
    '''
    return the lapacian stack and the lowest frequency gaussian image for reconstruciton, total layer is lv+1
    '''
    stackL = []
    # create a gaussian stack , require lv+1
    stackG = createGaussianStack(im,lv+1,sz)
    # create laplacian stack by subtraction
    for i in range(lv):
        stackL.append(stackG[i]-stackG[i+1])
    stackL.append(stackG[i+1].copy())
    return stackL


def collaspeStackL(stack):
    '''
    add all the original value from the Laplacian stack, user need to make sure the color range is correct for display
    '''
    sz = len(stack)
    if(sz==0):
        print("Empty stack!")
        return
    im = np.zeros(stack[0].shape)
    for s in stack:
        im+=s
    return im






######################### Image Alignment ##################################################


def computeCost(img1, img2, dx, dy):
    '''
    Shift img1 by dx, dy and compute the L2 norm of the difference between the overlap region of img1 and img2 
    '''
    #assert img1.shape==img2.shape
    h, w = img1.shape[0:2]
    # crop the image border, this make the process much faster and reduce the border's
    h_offset = int(h/4)
    w_offset = int(w/4)

    # compute overlap region of image1 and 2
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
    return np.linalg.norm(diff) # same as np.sqrt(np.sum(diff**2)), but faster


def generateImagePyramids(img, max_scale = 4):
    '''
    Generate image pyramids by halving the images and return the pyramids
    '''
    h, w = img.shape[0:2]
    # check maximum scale
    max_scale = int(min(np.floor(np.log2(h)),np.floor(np.log2(w)),max_scale)) 
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
    '''
    return the best alignment of img1 and img2 given an initial displacement dx, dy within a search_window
    '''
    assert img1.shape==img2.shape
    # convert to float
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
    # use 0.5 as image 2's BW threshold, compute the image 1 threshold
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

def findAlignmentMultiScale(pyramid1, pyramid2, scale, dx=0, dy=0, search_window = (-200,200)):
    '''
    return the best alignment of img1 and img2 (using the pyramids, in a recursive way), given an initial displacement dx, dy within a search_window
    '''
    # recursion guard
    if scale<0:
        return dx,dy   
    # input dx, dy are estimte from last scale, map them to current scale 
    dx*=2
    dy*=2
    # retrive image from pyramid
    img1 = pyramid1[scale]
    img2 = pyramid2[scale]
    best_dx, best_dy = findAlignmentSingleScale(img1,img2,dx,dy,search_window)
    return findAlignmentMultiScale(pyramid1, pyramid2, scale-1, best_dx, best_dy) # recursive call

def alignImagesSingleScale(img1,img2):
    '''
    call the single scale image aligning routine and return the aligned image (img1)
    '''
    best_dx, best_dy = findAlignmentSingleScale(img1, img2, 0,0,(-20,20))
    print(f'Displacement vector: ({best_dx,best_dy})')
    return np.array([[1.0,0.0,best_dx],[0.0,1.0,best_dy],[0.0,0.0,1.0]]).astype(np.float32)

def alignImagesMultiScale(pyramid1,pyramid2):
    '''
    call the multi scale image aligning routine and return the aligned image (img1)
    '''
    assert len(pyramid1)==len(pyramid2)
    scale = len(pyramid1)-1
    best_dx, best_dy = findAlignmentMultiScale(pyramid1, pyramid2, scale, 0,0,(-250,250))
    print(f'Displacement vector: ({best_dx,best_dy})')
    return  np.array([[1.0,0.0,best_dx],[0.0,1.0,best_dy],[0.0,0.0,1.0]]).astype(np.float32)


def computeT(im1,im2):
    '''
    return the translation matrix from im1 to im2
    '''
    pyd1 = generateImagePyramids(im1,8)
    pyd2 = generateImagePyramids(im2,8)
    return alignImagesMultiScale(pyd1,pyd2)
