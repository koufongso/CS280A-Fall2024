from dataclasses import dataclass

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import openpifpaf
import openpifpaf.predict

from VehicleKeyPointMap import *

# some constant
mps_to_mph = 2.237
(text_width, text_height), baseline = cv.getTextSize(f"Less than 4 keypoints   ", cv.FONT_HERSHEY_SIMPLEX, 0.5,thickness = 1)
ground_truth = 0
speed_output_record = []
need_rotate_cw = False

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

        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        dt = 1/fps #[s]
        output_size = (frame_width, frame_height)
        out = cv.VideoWriter("test_speed.avi", cv.VideoWriter_fourcc(*'MJPG'), fps, output_size)
        

        prefix = video_path.split(".")[0]
        im_prev = None
        count = 0
        count_detect = 0
        while True:
            count+=1

            ret, im = cap.read()

            if not ret:
                print("Video capture end. Exiting ...")
                break

            # map to rgb and resize image
            if need_rotate_cw:
                im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)

            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            (h,w,_) = im.shape
            
            # resize if needed 
            h = int(h/4)
            w = int(w/4)

            im_rz = cv.resize(im, (w, h))
            im_out = cv.cvtColor(im_rz, cv.COLOR_RGB2BGR)
            if im_prev is None:
                speed_output_record.append(None)
                # if first frame, skip
                im_prev = cv.cvtColor(im_rz, cv.COLOR_RGB2GRAY)
                cv.imshow('Frame', im_out)
                if cv.waitKey(1) & 0xFF == ord('s'): 
                    break
                continue
            
            # has previous frame, detect if vehicle exist
            predictions, _, _ = self.detector.numpy_image(im_rz)
            if len(predictions)==0:
              speed_output_record.append(None)
              print(f"{count}: no detect")
              cv.imshow('Frame', im_out)
              if cv.waitKey(1) & 0xFF == ord('s'): 
                  break
              continue

            # for now assume only one vehicle is detected
            if len(predictions)>1:
                print(f"{count}: Warning: more than 1 vehicle object detected!")
            else:
                print(f"{count}: Detected {len(predictions)} vehicle object.")

            # since we have vehicle detected, compute the optical flow now
            im_rz_gray = cv.cvtColor(im_rz, cv.COLOR_RGB2GRAY)
            im_prev_gaussian = cv.GaussianBlur(im_prev,(5,5),0)
            im_gaussian = cv.GaussianBlur(im_rz_gray,(5,5),0)
            flow = cv.calcOpticalFlowFarneback(im_prev_gaussian, im_gaussian, None, 0.5, 3, 20, 3, 5, 1.2, 0)
            
            im_warp = np.copy(im_rz) # for visualization
            
            for id, obj in enumerate(predictions): 
                im_out = self.annotateImage(im_out, obj)

                pts_candidates=[]
                index_list = []
                veh = VehicleKeyPointMap()
                # check keypoints of 1 vehicle object
                for idx,pt in enumerate(obj.data):
                    if pt[0]>0 and pt[1]>0 and pt[2]>0.7:
                        # valid keypoints
                        u = pt[0]
                        v = pt[1]
                        pts_candidates.append([u,v])
                        index_list.append(idx)

                # get the keypoints that belongs to left/right side of the vehicle
                kpt_index_list, pt_src, pt_des, info =  veh.getCorrespondenceCoord(index_list,pts_candidates,h,w)    
                if(pt_des.shape[0]>=4):
                    H,_ = cv.findHomography(pt_src, pt_des)
                else:
                    print(f"{count}: Less than 4 valid keypoints detected: number of valid keypoint(s):{len(pt_src)}")
                    im_prev = None # no valid car detect in the current frame, drop the previous frame as well
                    speed_output_record.append(None)
                    im_out = self.annotateImage(im_out, obj, pt_src)
                    cv.imshow('Frame', im_out)
                    if cv.waitKey(1) & 0xFF == ord('s'): 
                        break
                    continue

                # query the keypoints' flow
                pt_src_tail = [] # this the is the vecotr arrow "head": (pt_src) x-----> (head/pt_src_tail)
                for pt in pt_src:
                    x0 = pt[0]
                    y0 = pt[1]
                    v = flow[int(y0),int(x0)]
                    vx = v[0]
                    vy = v[1]
                    x1 = x0+vx
                    y1 = y0+vy
                    pt_src_tail.append([x1,y1])

                    # add optical flow vector
                    cv.arrowedLine(im_out, (x0, y0), (x1,y1), (0, 0, 255), 1, tipLength=0.1)

                n = pt_des.shape[0]
                pt_warp = H @ np.hstack((pt_src,np.ones((n,1)))).T
                pt_tail_warp = H @ np.hstack((pt_src_tail,np.ones((n,1)))).T

                pt_warp /= pt_warp[2,:]
                pt_tail_warp /= pt_tail_warp[2,:]

                speed_warp = info.factor_mpp* (pt_tail_warp - pt_warp)/dt * mps_to_mph
                speed_x = speed_warp[0,:]
                speed_final = abs(np.mean(speed_x))
                #print(f"average speed = {speed_final:.1f} MPH")
                im_out = self.annotateImage(im_out,obj,pt_src, speed_final)
                speed_output_record.append(speed_final)
                
                # save intermediate result
                # car_bbox+speed+keypoints
                cv.imshow('Frame', im_out)
                if cv.waitKey(1) & 0xFF == ord('s'): 
                    break


                # save the annotated original image
                cv.imwrite(f'{count:04d}.jpg', im_out)

                # save the warp image
                im_warp = cv.warpPerspective(im_warp, H, (w, h))
                im_warp = cv.cvtColor(im_warp, cv.COLOR_RGB2BGR)
                for p0, p1 in zip(pt_warp.T,pt_tail_warp.T):
                    x0 = int(p0[0])
                    y0 = int(p0[1])
                    x1 = int(p1[0])
                    y1 = int(p1[1])
                    cv.circle(im_warp, (x0, y0), radius=2, color=(255, 0, 0), thickness=-1)
                    cv.arrowedLine(im_warp, (x0, y0), (x1,y1), (0, 0, 255), 1, tipLength=0.1)
                im_warp = cv.flip(im_warp,1)
                cv.imwrite(f'w_{count:04d}.jpg', im_warp)
                
            im_prev = im_rz_gray
            count_detect+=1

            # if count_detect>20:
            #     break

        # When everything done, release the capture
        cap.release()
        #out.release()
        cv.destroyAllWindows()

        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(speed_output_record)),speed_output_record,color='b', label='Estimation')
        plt.axhline(y = ground_truth, color = 'r', label = 'Speed gun measurement')
        plt.xlabel('Frame')
        plt.ylabel('Speed [MPH]')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"est_speed.jpg")
            
    def annotateImage(self, im, obj=None, kpts=None, speed = None):
        # draw bounding box
        if obj is not None:
            corner =obj.bbox()
            x1 = int(corner[0])
            y1 = int(corner[1])
            x2 = int(x1+corner[2])
            y2 = int(y1+corner[3])
            cv.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv.rectangle(im, (x1, y1), (x1+text_width, y1-text_height), (0, 0, 0), thickness=-1)
            if speed is not None:
                cv.putText(im,f"Speed {speed:.1f} MPH",(x1, y1),cv.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255),thickness = 1)
            else:
                cv.putText(im,f"Less than 4 keypoints",(x1, y1),cv.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255),thickness = 1)

        if kpts is not None:
            for kpt in kpts:
                x = kpt[0]
                y = kpt[1]
                cv.circle(im, (x, y), radius=2, color=(255, 0, 0), thickness=-1)


        return im


if __name__ == "__main__":
    video_path = "./final_project/dataset/videos/IMG_4960.MOV"
    ground_truth = 17.0
    need_rotate_cw = False
    estimator = SpeedEstimator()
    estimator.run(video_path)