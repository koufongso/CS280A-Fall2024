from dataclasses import dataclass

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import openpifpaf
import openpifpaf.predict

from VehicleKeyPointMap import *

mps_to_mph = 2.237

class SpeedEstimator:
    def __init__(self) -> None:
        self.use_video_stream = False
        self.use_video_record = True
        self.detector = openpifpaf.Predictor(checkpoint='shufflenetv2k16-apollo-66')

    def run(self, video_path):
        assert(not(self.use_video_record and self.use_video_stream))
        if self.use_video_stream:
            cap = cv.VideoCapture(0)
        elif self.use_video_record:
            cap = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video capture.")
            exit()

        # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        dt = 1/fps #[s]
        # fourcc = int(cap.get(cv.CAP_PROP_FOURCC))  # Codec used in the original video
        # output_size = (frame_width, frame_height)
        # out = cv.VideoWriter(video_path+"_warp.MOV", cv.VideoWriter_fourcc(*'mp4v'), fps, output_size)


        prefix = video_path.split(".")[0]
        im_warp_prev = None
        count = 0
        count_detect = 0
        while True:
            count+=1

            ret, im = cap.read()

            if not ret:
                print("Video capture end. Exiting ...")
                break

            # process image here:
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            (h,w,_) = im.shape
            
            # resize if needed 
            h = int(h/4)
            w = int(w/4)

            im_rz = cv.resize(im, (w, h))
            predictions, _, _ = self.detector.numpy_image(im_rz)
            if len(predictions)==0:
              print(f"{count}: no detect")
              continue

            print(f"{count}: detect")
            for id, obj in enumerate(predictions): # for now assume only one vehicle is detected
                pts=[]
                index_list = []
                veh = VehicleKeyPointMap()
                for idx,pt in enumerate(obj.data):
                    if pt[0]>0 and pt[1]>0:
                        # valid keypoints
                        u = pt[0]
                        v = pt[1]
                        pts.append([u,v])
                        index_list.append(idx)

                # get the keypoints that belongs to left/right side of the vehicle
                kpt_index_list, pt_src, pt_des, info =  veh.getCorrespondenceCoord(index_list,pts,h,w)
                # for visualization
                for i, pt in zip(kpt_index_list, pt_src):
                    x = int(pt[0])
                    y = int(pt[1])
                    #im_tmp = cv.circle(im_rz, (x, y), 5, (0, 255, 0), -1)
                    #%im_tmp = cv.putText(im_rz,f"{i}",(x + 10, y + 10),cv.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0),thickness = 1)
                
                if(pt_des.shape[0]>=4):
                    H,_ = cv.findHomography(pt_src, pt_des)
                    im_warp = cv.warpPerspective(im_rz, H, (w, h))
                    im_warp = cv.cvtColor(im_warp, cv.COLOR_RGB2GRAY)

                if im_warp_prev is None:
                    im_warp_prev = im_warp
                    continue

                if im_warp is None:
                    im_warp_prev = None
                    continue

                # im_warp_prev and im_warp exist
                im_warp_prev_gaussian = cv.GaussianBlur(im_warp_prev,(5,5),0)
                im_warp_gaussian = cv.GaussianBlur(im_warp,(5,5),0)


                flow = cv.calcOpticalFlowFarneback(im_warp_prev_gaussian, im_warp_gaussian, None, 0.5, 3, 20, 3, 5, 1.2, 0)
                #magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
                flow_x = flow[...,0]
                # only keep the bottom part (near the wheels)
                # from dx:dx+w, dy:dy+
                lx = int(info.dx+30)
                ux = int(info.dx+info.w-30)
                ly = int(info.dy+20)
                uy = int(info.dy+50)

                flow_x_roi = flow_x[ly:uy,lx:ux]
                #flow_roi = magnitude[magnitude]
                #print(info.factor_mpp,dt,mps_to_mph)
                speed_x = info.factor_mpp *  flow_x_roi/ dt * mps_to_mph   # m/s
                speed_x = speed_x.flatten()
                
                hist, bin_edges = np.histogram(speed_x, bins=50, range=(0, 100))
                smoothed_hist = gaussian_filter1d(hist,0.5)
                peaks, _ = find_peaks(smoothed_hist, prominence=10)

                plt.figure(figsize=(8, 6))
                plt.plot(bin_edges[:-1], hist, label='Original Histogram', alpha=0.6)
                plt.plot(bin_edges[:-1], smoothed_hist, label='Smoothed Histogram', alpha=0.6)
                plt.scatter(bin_edges[peaks], smoothed_hist[peaks], color='red', label='Detected Peaks')
                plt.axvline(x = 29, color = 'g', label = 'speed gun measurement')
                plt.title('Histogram of X-Direction Optical Flow')
                plt.xlabel('MPH')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{prefix}_{count:05d}_of_hist.jpg")


                step = 5
                output = im_warp
                for y in range(ly, uy, step):
                    for x in range(lx, ux, step):
                        # Extract the flow vector at (x, y)
                        dx, dy = flow[y, x]    
                        # Scale the vector for better visualization
                        end_point = (int(x + dx), int(y))     
                        # Draw the arrowed line on the image
                        cv.arrowedLine(output, (x, y), end_point, (0, 255, 0), 1, tipLength=0.1)
                cv.imwrite(f"{prefix}_{count:05d}_of.jpg", output)
                
                im_warp_prev = im_warp

                break # assume one car for now

            # # show the resulting frame
            # #im_rz = cv.cvtColor(im_rz, cv.COLOR_RGB2BGR)
            # im_warp = cv.cvtColor(im_warp, cv.COLOR_RGB2BGR)
            # #cv.imwrite(f"{prefix}_{count:05d}.jpg",im_rz)
            # cv.imwrite(f"{prefix}_{count:05d}_w.jpg",im_warp)
            count_detect+=1

            if count_detect>20:
                break
            





        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()





if __name__ == "__main__":
    video_path = "./final_project/dataset/videos/IMG_4938.MOV"
    estimator = SpeedEstimator()
    estimator.run(video_path)