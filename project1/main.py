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

########################################## Bonus/ improvment#############################################
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

def findImageBoarder(img, threshold1 = 10000, threshold2 = 25):
    '''
    find the input image board and return the index of the boarder by filter vecotr (colmuns/rows) L2 cost and std with given threshold
    '''
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grey_img = cv.GaussianBlur(grey_img, (7, 7), 0) # remove noise 
    #edges = cv.Canny(grey_img,50,50) # directly using grey scale image seems has better performance finding boarder
    
    h,w = grey_img.shape

    # searching window: left [0,x0] right [x1,end] top [0,y0] bot [y1,end]
    x0 = int(w/6)
    x1 = w-x0
    y0 = int(h/6)
    y1 = h-y0

    # search left
    j = x0
    col_prev = grey_img[:,j]
    j-=1
    while(j>0):
        col = grey_img[:,j]
        diff = LA.norm(col-col_prev)
        if(diff<threshold1 and np.std(col)<threshold2):
            break
        col_prev = col
        j-=1
    left = j
    # search right
    j = x1
    col_prev = grey_img[:,j]
    j-=1
    while(j<w-1):
        col = grey_img[:,j]
        diff = LA.norm(col-col_prev)
        if(diff<threshold1 and np.std(col)<threshold2):
            break
        col_prev = col
        j+=1
    right = j
    # search top
    i = y0
    row_prev = grey_img[i,:]
    i-=1
    while(i>0):
        row = grey_img[i,:]
        diff = LA.norm(row-row_prev)
        if(diff<threshold1 and np.std(row)<threshold2):
            break
        row_prev = row
        i-=1
    top = i
    # search bot
    i = y1
    row_prev = grey_img[i,:]
    i+=1
    while(i<h-1):
        row = grey_img[i,:]
        diff = LA.norm(row-row_prev)
        if(diff<threshold1 and np.std(row)<threshold2):
            break
        row_prev = row
        i+=1
    bot = i

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
    # use colorized images for fast debugging
    filepath_ms.append("./images_colorized/cathedral_c.jpg")
    filepath_ms.append("./images_colorized/monastery_c.jpg")
    filepath_ms.append("./images_colorized/tobolsk_c.jpg")
    filepath_ms.append("./images_colorized/lady_c.jpg")
    filepath_ms.append("./images_colorized/train_c.jpg")
    filepath_ms.append("./images_colorized/church_c.jpg")
    filepath_ms.append("./images_colorized/emir_c.jpg")
    filepath_ms.append("./images_colorized/harvesters_c.jpg")
    filepath_ms.append("./images_colorized/icon_c.jpg")
    filepath_ms.append("./images_colorized/melons_c.jpg")
    filepath_ms.append("./images_colorized/onion_church_c.jpg")
    filepath_ms.append("./images_colorized/sculpture_c.jpg")
    filepath_ms.append("./images_colorized/self_portrait_c.jpg")
    filepath_ms.append("./images_colorized/three_generations_c.jpg")
    for filepath in filepath_ms:
        # single scale
        t0 = time.time()
        filename = filepath.split('/')[-1].split('.')[0]
        print(filename)
        # load image
        image = cv.imread(filepath)
        if(image is None):
            print(f"Cannot open file: {filepath}")
            exit(-1)

        # combine = np.dstack((b,ag,ar))
        combine = cropImageBorders(image)
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
        # run main function, need to provide input output path
        main(sys.argv[1],sys.argv[2])
    else:
        # run the test functions

        # please modify the input filepath list and the save path (cv.imwrite) in these test functions
        print("No input arguments, uncomment the test function to run test, make sure the file paths within the functions are correct.")
       
        #test_single_scale()
       
        #test_multi_scale()
        
        #test_enhanced()