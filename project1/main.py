import cv2 as cv
import numpy as np
from numpy import linalg as LA
import time
import sys

############################# Functions ############################################
'''
There are 10 functions
For the core homework:
1. cropImageBGR
2. computeCost
3. generateImagePyramids
4. findAlignmentSingleScale
5. findAlignmentMultiScale
6. alignImagesSingleScale
7. alignImagesMultiScale

For the bounus part (photo improment/enhancement)
8. colorMapping
9. findImageBoarder
10. cropImageBorders

'''

def cropImageBGR(img):
    '''
    Just split image into three channel at 1/3 and 2/3 height
    '''
    height = img.shape[0]
    h = int(np.floor(height/3.0))
    return img[:h], img[h:2*h], img[2*h:3*h]

def computeCost(img1, img2, dx, dy):
    '''
    Shift img1 by dx, dy and compute the L2 norm of the difference between the overlap region of img1 and img2 
    '''
    #assert img1.shape==img2.shape
    h, w = img1.shape
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
    return LA.norm(diff) # same as np.sqrt(np.sum(diff**2)), but faster


def generateImagePyramids(img, max_scale = 4):
    '''
    Generate image pyramids by halving the images and return the pyramids
    '''
    h, w = img.shape
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

def findAlignmentMultiScale(pyramid1, pyramid2, scale, dx=0, dy=0, search_window = (-15,15)):
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
    best_dx, best_dy = findAlignmentSingleScale(img1, img2, 0,0,(-15,15))
    h,w = img2.shape
    print(f'Displacement vector: ({best_dx,best_dy})')
    return cv.warpAffine(img1, np.array([[1,0,best_dx],[0,1,best_dy]]).astype(np.float32), (w,h))


def alignImagesMultiScale(pyramid1,pyramid2):
    '''
    call the multi scale image aligning routine and return the aligned image (img1)
    '''
    assert len(pyramid1)==len(pyramid2)
    scale = len(pyramid1)-1
    best_dx, best_dy = findAlignmentMultiScale(pyramid1, pyramid2, scale, 0,0,(-15,15))
    h,w = pyramid2[0].shape
    print(f'Displacement vector: ({best_dx,best_dy})')
    return cv.warpAffine(pyramid1[0], np.array([[1,0,best_dx],[0,1,best_dy]]).astype(np.float32), (w,h))


def colorMapping(img):
    '''
    map the red, green, blue filter (passing) image (input as BGR order) to the BGR color
    '''
    h,w,_ = img.shape
    M = np.array([[0.8790,    0.3653,   -0.2751],[-0.1064,    1.0121,    0.0700],[-0.5003,    0.4384,    1.0365]]) # color transformation matrix (BGR order!)
    img_ = img.reshape(-1,3).transpose()        # convert image to [3,w*h] shape
    img_ = M @ img_                             # map the color as new_BRG = M * old_BRG
    img_ = img_.transpose().reshape(h,w,3)      # convert back to original shape
    return img_

def findImageBoarder(img, threshold = 0.9, offset = 2):
    '''
    find the input image board and return the index of the boarder by compare the std of the rows/cols of the image and check if the ratio pass certain threshold
    '''
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.uint8)
    grey_img = cv.GaussianBlur(grey_img, (5, 5), 0) # remove noise 
    #edges = cv.Canny(grey_img,50,50) # directly using grey scale image seems has better performance finding boarder
    
    img_ = grey_img
    h,w = img_.shape
    
    # return variables
    top = 0
    bot = h
    left = 0
    right = w

    # starting position
    x1 = int(w/2)   # left 
    x2 = x1+1       # right
    y1 = int(h/2)   # top 
    y2 = y1+1       # bot

    # left 
    line = img_[int(h/6):int(5*h/6),max(0,x1-offset):min(w,x1+offset+1)]
    var_prev = np.std(line)
    while(x1>0):
        x1-=1
        line = img_[int(h/6):int(5*h/6),max(0,x1-offset):min(w,x1+offset+1)]
        var = np.std(line)
        if(var/var_prev<threshold and np.mean(line)<120):
            left = x1
            break
        var_prev = var
        
    # right
    line = img_[int(h/6):int(5*h/6),max(0,x2-offset):min(w,x2+offset+1)]
    var_prev = np.std(line)
    while(x2<w):
        x2+=1
        line = img_[int(h/6):int(5*h/6),max(0,x2-offset):min(w,x2+offset+1)]
        var = np.std(line)
        if(var/var_prev<threshold and np.mean(line)<120):
            right = x2
            break
        var_prev = var

    # top
    line = img_[max(0,y1-offset):min(h,y1+offset+1),int(w/6):int(5*w/6)]
    var_prev =np.std(line)
    while(y1>0):
        y1-=1
        line = img_[max(0,y1-offset):min(h,y1+offset+1),int(w/6):int(5*w/6)]
        var = np.std(line)
        if(var/var_prev<threshold and np.mean(line)<120):
            top = y1
            break
        var_prev = var
             
    # bot
    line = img_[max(0,y2-offset):min(h,y2+offset+1),int(w/6):int(5*w/6)]
    var_prev = np.std(line)
    while(y2<h):
        y2+=1
        line = img_[max(0,y2-offset):min(h,y2+offset+1),int(w/6):int(5*w/6)]
        var = np.std(line)
        if(var/var_prev<threshold and np.mean(line)<120):
            bot = y2
            break
        var_prev = var
        
    return left, right, top, bot


def cropImageBorders(img):
    '''
    find the image boarder and return the cropped image
    '''
    x1, x2, y1, y2 = findImageBoarder(img)
    print(x1,x2,y1,y2)
    return img[y1:y2,x1:x2]


############################# Testing scripts ############################################

def test_single_scale():
    filepath_ss=[]
    # small size, single scale
    filepath_ss.append("./images/cathedral.jpg")
    filepath_ss.append("./images/monastery.jpg")
    filepath_ss.append("./images/tobolsk.jpg")
    for filepath in filepath_ss:
        # single scale
        t0 = time.time()
        filename = filepath.split('/')[-1].split('.')[0]
        print(filename)
        # load image
        image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if (image is None):
            print(f"Cannot open file: {filepath}")
            exit(-1)
        
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
    filepath_ms.append("./images/lady.tif")
    filepath_ms.append("./images/train.tif")
    filepath_ms.append("./images/church.tif")
    filepath_ms.append("./images/emir.tif")
    filepath_ms.append("./images/harvesters.tif")
    filepath_ms.append("./images/icon.tif")
    filepath_ms.append("./images/melons.tif")
    filepath_ms.append("./images/onion_church.tif")
    filepath_ms.append("./images/sculpture.tif")
    filepath_ms.append("./images/self_portrait.tif")
    filepath_ms.append("./images/three_generations.tif")

    for filepath in filepath_ms:
        t0 = time.time()
        filename = filepath.split('/')[-1].split('.')[0]
        print(filename)

        # load image
        image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if (image is None):
            print(f"Cannot open file: {filepath}")
            exit(-1)
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


def test_enhanced():
    filepath_ms=[]
    filepath_ms.append("./images/cathedral.jpg")
    filepath_ms.append("./images/monastery.jpg")
    filepath_ms.append("./images/tobolsk.jpg")
    filepath_ms.append("./images/lady.tif")
    filepath_ms.append("./images/train.tif")
    filepath_ms.append("./images/church.tif")
    filepath_ms.append("./images/emir.tif")
    filepath_ms.append("./images/harvesters.tif")
    filepath_ms.append("./images/icon.tif")
    filepath_ms.append("./images/melons.tif")
    filepath_ms.append("./images/onion_church.tif")
    filepath_ms.append("./images/sculpture.tif")
    filepath_ms.append("./images/self_portrait.tif")
    filepath_ms.append("./images/three_generations.tif")
    for filepath in filepath_ms:
        # single scale
        t0 = time.time()
        filename = filepath.split('/')[-1].split('.')[0]
        print(filename)
        # load image
        image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if(image is None):
            print(f"Cannot open file: {filepath}")
            exit(-1)
        # crop image
        b,g,r = cropImageBGR(image)

        # create pyramids     
        b_pyd = generateImagePyramids(b,4)
        g_pyd = generateImagePyramids(g,4)
        r_pyd = generateImagePyramids(r,4)
        # align image, multi scale
        ag = alignImagesMultiScale(g_pyd,b_pyd)
        ar = alignImagesMultiScale(r_pyd,b_pyd)

        combine = np.dstack((b,ag,ar))
        combine = cropImageBorders(combine)
        combine = colorMapping(combine)

        cv.imwrite(f"./images_enhanced/{filename}_e.jpg",combine)
        t1 = time.time()
        print(f'Total elapsed time: {t1-t0:.3}s\n')


############################# Main ############################################
def main(fin, fout):
    t0 = time.time()
    print(f'Open file: {fin}')
    image = cv.imread(fin, cv.IMREAD_GRAYSCALE)
    if(image is None):
        print(f"Cannot open file: {fin}")
        exit(-1)
    
    # crop image
    b,g,r = cropImageBGR(image)

    # create pyramids     
    b_pyd = generateImagePyramids(b,4)
    g_pyd = generateImagePyramids(g,4)
    r_pyd = generateImagePyramids(r,4)
    # align image, multi scale
    ag = alignImagesMultiScale(g_pyd,b_pyd)
    ar = alignImagesMultiScale(r_pyd,b_pyd)

    combine = np.dstack((b,ag,ar))
    combine = cropImageBorders(combine)
    combine = colorMapping(combine)
    
    res = cv.imwrite(f"{fout}",combine)
    if(res):
        print(f'image saved.')
    else:
        print(f'faile to save image.') 
    t1 = time.time()
    print(f'Total elapsed time: {t1-t0:.3}s\n')
    

if __name__ == "__main__":
    if( len(sys.argv) > 2):
        # run main function
        main(sys.argv[1],sys.argv[2])
    else:
        # run the test functions
        # please modify the input filepath list and the save path (cv.imwrite) in these test functions
        print("No input arguments, run testing functions! Might take 15 min to run all 3 tests.")
        print("-"*20)
        print("Test single scale")
        #test_single_scale()
        print("-"*20)
        #print("Test multi scale")
        #test_multi_scale()
        print("-"*20)
        print("Test enhanced")
        test_enhanced()