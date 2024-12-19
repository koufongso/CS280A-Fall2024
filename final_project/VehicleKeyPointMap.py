import cv2 as cv
import numpy as np
# Mapping of the 66 car keypoint image/pixel coordinate
# self.coord store the pixel coordinate of the keypoint of the corresponding index in the format of (x,y), or (col, row) order

from dataclasses import dataclass

@dataclass
class VehicleKeyPointInfo:
    "the transformation about the base keypoint model"
    def __init__(self, scale, dx, dy, h, w, factor_mpp):
        self.scale: float = scale
        self.dx: int = dx
        self.dy:int = dy
        self.h: int = h                     # keypoint model bounding box after scaling
        self.w: int = w                     # keypoint model bounding box after scaling
        self.factor_mpp: float = factor_mpp # meter per pixel

    def print(self):
        print(f"scale: {self.scale}")
        print(f"(dx,dy): {(self.dx, self.dy)}")
        print(f"(h,w): {(self.h, self.w)}")



class VehicleKeyPointMap:
    def __init__(self):
        self.coord = [[np.nan,np.nan]]*66

        # index to retieve left/right face keypoint from the coord
        self.left_idx = [6,7,10,15,19,20,21,13,14,17,18,9,12]
        self.is_left_kpt = [False]*66
        for idx in self.left_idx:
            self.is_left_kpt[idx] = True

        self.right_idx = [51,50,47,42,38,37,36,44,43,40,39,48,45]
        self.is_right_kpt = [False]*66
        for idx in self.right_idx:
            self.is_right_kpt[idx] = True

        # left side
        self.coord[6] = [0,0]
        self.coord[7] = [222,-88]
        self.coord[10]= [599,-53]
        self.coord[15]= [1250,-53]
        self.coord[19]= [1609,-86]
        self.coord[20]= [1951,-88]
        self.coord[21]= [2192,-46]
        self.coord[13]= [1081,-388]
        self.coord[14]= [1207,-388]
        self.coord[17]= [1646,-388]
        self.coord[18]= [1769,-388]
        self.coord[9]= [617,-468]
        self.coord[12]= [1244,-483]
        self.coord[16]= [1661,-528]
        self.coord[22]= [2356,-438]
        self.coord[23]= [2374,-363]


        # right side
        self.coord[51] = self.coord[6]
        self.coord[50] = self.coord[7]
        self.coord[47] = self.coord[10]
        self.coord[42] = self.coord[15]
        self.coord[38] = self.coord[19]
        self.coord[37] = self.coord[20]
        self.coord[36] = self.coord[21]
        self.coord[44] = self.coord[13]
        self.coord[43] = self.coord[14]
        self.coord[40] = self.coord[17]
        self.coord[39] = self.coord[18]
        self.coord[48] = self.coord[9]
        self.coord[45] = self.coord[12]
        self.coord[41] = self.coord[16]
        self.coord[35] = self.coord[22]
        self.coord[34] = self.coord[23]

        self.coord = np.array(self.coord)

        self.h0 = 550
        self.w0 = 2400
        self.d = 3.0 # distance between wheels
        self.factor_m_p = self.d /(self.coord[20][0]-self.coord[7][0]) # approx [m/px]

    def getCoord(self ,scale = 1.0, offset = [100,800]):
        return scale* self.coord + offset
    
    def getCorrespondenceCoord(self, index_list, coord_list, h, w):
        sx = (0.8*w)/self.w0    # leave 10% margin on top and bottom
        sy =(0.8*h)/self.h0     
        scale = min(sx,sy)
        dx = (w - scale*self.w0)/2
        dy = (h + scale*self.h0)/2

        info = VehicleKeyPointInfo(scale, dx, dy, scale*self.h0, scale*self.w0, self.factor_m_p/scale)

        map_kpt = self.getCoord(scale,[dx,dy])

        group_index = None
        is_kpt = None
        
        pt_src = [] # source points/ original frame
        pt_des = [] # target points/ vehicle keypoint template
        matched_idx=[]

        for idx, pt in zip(index_list,coord_list):
            #
            if group_index is None:
                if(self.is_left_kpt[idx]):
                    #print("use vehicle left side keypoints")
                    group_index = self.left_idx  
                    is_kpt = self.is_left_kpt
                if(self.is_right_kpt[idx]):
                    #print("use vehicle right side keypoints")
                    group_index = self.right_idx
                    is_kpt = self.is_right_kpt
            
            if is_kpt is not None:
                if(is_kpt[idx]):
                    pt_src.append(pt)
                    pt_des.append(map_kpt[idx])
                    matched_idx.append(idx)

        return matched_idx, np.array(pt_src).astype(np.float32), np.array(pt_des).astype(np.float32), info
            


    def visualize(self, h, w, group = "left"):
        sx = (0.8*w)/self.w0    # leave 10% margin on top and bottom
        sy =(0.8*h)/self.h0     
        scale = min(sx,sy)
        dx = (w - scale*self.w0)/2
        dy = (h + scale*self.h0)/2

        coord_tmp = self.getCoord(scale,offset=[dx,dy])
        image = np.zeros((h, w, 3), dtype=np.uint8)

        if group == "left":
            group_index = self.left_idx
        elif group == "right":
            group_index = self.right_idx
        else:
            ValueError(f"[visualization]: Aviable group:'left', 'rigth', but input is {group}. ")
            return

        for point, idx in zip(coord_tmp[group_index],group_index):
            x,y,=point
            x = int(x)
            y = int(y)
            cv.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # Green circles
            cv.putText(image,f"{idx}",(x + 10, y + 10),cv.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0),thickness = 1)

        # Display the image
        cv.imshow("Car Keypoint", image)
        cv.waitKey(0)
        cv.destroyAllWindows()




if __name__ == "__main__":
    veh1 = VehicleKeyPointMap()
    veh1.visualize(960,540)


