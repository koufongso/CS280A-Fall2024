import cv2 as cv
#import matplotlib.pyplot.ginput as ginput
import pickle

from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory

import numpy as np

# utility function, use to load and convert correspondence data to np.array

def loadCorrespondence(filepath:str):
    '''
    read the correspondence pair from the pickle file. the pair is  a dictionary with {id, Correspondence object} pair
    '''
    print(f"load correspondence pari from {filepath}")
    with open(filepath,'rb') as file:
        pair = pickle.load(file)
        print("Correspondence pair loaded!")
        file.close()
    return pair

def coordsFromPairs(pair):
    '''
    Extract coordinates from a list of Correspondence pairs into two 2D arrays.
    '''
    pts1 = []
    pts2 = []

    for key, c in pair.items():
        pts1.append([c.obj1.x,c.obj1.y])
        pts2.append([c.obj2.x,c.obj2.y])
    
    return np.array(pts1), np.array(pts2)


class Point2:
    '''
    2D point data struct
    '''
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y
        

class Correspondence:
    '''
    correspondence data struct
    '''
    id = 0
    def __init__(self, obj1, obj2, name:str = None) -> None:
         # make sure store the ame type of object
        if not isinstance(obj1, type(obj2)):
            raise TypeError("Objects in correspondence are not the same type!")
        
        # manage propety
        self.type = type(obj1)
        Correspondence.id+=1
        self.id = Correspondence.id
        self.name = name if name is not None else f'correspondence_{self.id}'
        
        # store correspondence pair
        self.obj1 = obj1
        self.obj2 = obj2        

class InternalData:
    def __init__(self) -> None:
        self.im1 = None
        self.im2 = None
        self.pair = {}

class AnnotationTool:
    def __init__(self) -> None:
        self.data = InternalData()

        # some GUI function "mutex"
        self.inuse_removeCorrespondenceGUI = False
        self.inuse_addCorrespondenceGUI = False

    # add/remove of images and correspondence 
    def loadImage(self, image_path, idx:int):
        im = mpimg.imread(image_path)
        if idx == 1:         
            self.data.im1 = im
        elif idx ==2:
            self.data.im2 = im
        else:
            print("loadImage: image index must less than 2!")
        
        if(self.data.im1 is not None and self.data.im2 is not None and self.data.im1.shape != self.data.im2.shape):
            print("Warning: two images size are not the same.")
    
    def resetImage(self, idx:int):
        if idx == 1:
            self.data.im1 = None
        elif idx == 2:
            self.data.im2 = None
        else:
            print("resetImage: image index must less than 2!")

    def addCorrespondence(self,c:Correspondence):
        self.data.pair[c.id] = c

    def removeCorrespondence(self,id:int):
        if id in self.data.pair:
            self.data.pair.pop(id)
        else:
            print(f"id {id} does not exist.")

    def resetCorrespondence(self):
        self.data.pair.clear()

    def reset(self):
        self.resetImage(1)
        self.resetImage(2)
        self.resetCorrespondence()

    def outputTXT(self,folder_path:str = None):
        if len(self.data.pair)==0:
             print("No data to save.")
             return

        filename1="/points1.txt"
        filename2="/points2.txt"
        if folder_path is not None:
            filename1 = folder_path+filename1
            filename2 = folder_path+filename2

        pts1,pts2 = coordsFromPairs(self.data.pair)
        
        np.savetxt(filename1, pts1)
        np.savetxt(filename2, pts2)

        print("Correspondence pair outputed as txt files!")

    def saveInternalData(self,filename:str = None):
        # if len(self.data.pair)==0:
        #     print("No data to save.")
        #     return
        
        if filename is None:
            filename = "data.pkl"

        print(f"Save data to {filename}")
        with open(filename,'wb') as file:
            pickle.dump(self.data, file)
            print("Data saved!")
            file.close()

    def loadInternalData(self,filepath:str):
        print(f"Load data {filepath}")
        with open(filepath,'rb') as file:
            self.data = pickle.load(file)
            print("Data loaded!")
            file.close()

        # need to update Correspondence id
        keys = self.data.pair.keys()
        if(len(keys)!=0):
            Correspondence.id = max(keys)
            

    ######################################## use for GUI #############################################

    def displayImage(self, idx, ax, canvas):
        if(idx==1):
            im = self.data.im1
        elif (idx==2):
            im = self.data.im2
        else:
            print("displayImage: image index must less than 2!")
            return
        if(im is None):
            print("displayImage: no image to display!")
            return
        
        ax.imshow(im)
        ax.axis('off')  # Hide axes for clean image display
        canvas.draw()

    def addCorrespondenceGUI(self, ax1, ax2, canvas1, canvas2, listbox, plt_ref, msg):
        if self.inuse_addCorrespondenceGUI == True:
            return
        
        self.inuse_addCorrespondenceGUI = True
        if(self.data.im1 is None or self.data.im2 is None):
            print("Not enought images.")
            msg.config(text="Message: Not enought images.")
            self.inuse_addCorrespondenceGUI = False
            return
        
        # point 1 from image 1
        plt.sca(ax1)
        msg.config(text="Message: Pick one point on image 1.")
        point = plt.ginput(1)
        if not point:
            print("Timeout")
            msg.config(text="Message: Timeout, please click add pair agian.")
            self.inuse_addCorrespondenceGUI = False
            return
        x1, y1 = point[0]
        point1, = ax1.plot(x1, y1, 'ro', markersize = 3)
        canvas1.draw()
        msg.config(text="Message: Pick one point on image 2.")
        # point 2 from image 2
        plt.sca(ax2)
        point = plt.ginput(1)
        if not point:
            print("Timeout")
            point1.remove()
            msg.config(text="Message: Timeout, please click add pair agian.")
            self.inuse_addCorrespondenceGUI = False
            return
        x2, y2 = point[0]
        point2, = ax2.plot(x2, y2, 'ro', markersize = 3)
        canvas2.draw()

        # add
        self.addCorrespondence(Correspondence(Point2(x1,y1),Point2(x2,y2)))
        id = Correspondence.id
        listbox.insert(id,f'Pair {id}')

        # update image
        text1 = ax1.text(x1, y1, f'{id}', fontsize=10, ha='right')
        text2 = ax2.text(x2, y2, f'{id}', fontsize=10, ha='right')
        canvas1.draw()
        canvas2.draw()

        # store points and text obj
        plt_ref["points1"][id] = point1
        plt_ref["points2"][id] = point2
        plt_ref["texts1"][id] = text1
        plt_ref["texts2"][id] = text2

        self.inuse_addCorrespondenceGUI = False
        msg.config(text="Message:")

    def removeCorrespondenceGUI(self,listbox, canvas1, canvas2, plt_ref):
        if self.inuse_removeCorrespondenceGUI== True:
            return
        self.inuse_removeCorrespondenceGUI = True
        idx = listbox.curselection()
        if(not idx):
            print("No item selected!")
            self.inuse_removeCorrespondenceGUI = False
            return
        idx = idx[0]
        name = listbox.get(idx)
        id = int(name.split()[-1])
        listbox.delete(idx)
        self.removeCorrespondence(id)

        # remove points and text on the images
        plt_ref["points1"][id].remove()
        plt_ref["points1"].pop(id)

        plt_ref["points2"][id].remove()
        plt_ref["points2"].pop(id)

        plt_ref["texts2"][id].remove()
        plt_ref["texts2"].pop(id)

        plt_ref["texts1"][id].remove()
        plt_ref["texts1"].pop(id)      
        
        canvas1.draw()
        canvas2.draw()

        self.inuse_removeCorrespondenceGUI = False

    def outputTXTGUI(self):
        folder_path = askdirectory()
        if(len(folder_path)!=0):
            self.outputTXT(folder_path)

    def saveInternalDataGUI(self):
        filename = asksaveasfilename(defaultextension=".pkl",filetypes=(("pickle file", "*.pkl"),("All Files", "*.*")))
        if(len(filename)!=0):
            self.saveInternalData(filename)

    def loadInternalDataGUI(self,listbox, ax1, ax2, canvas1, canvas2, plt_ref):
        filename = askopenfilename()
        if(len(filename)!=0):
            self.loadInternalData(filename)

            # update canvas, listbox, plt_ref
            self.displayImage(1,ax1,canvas1)
            self.displayImage(2,ax2,canvas2)

            for id, c in self.data.pair.items():
                # update listbox
                listbox.insert(id,f'Pair {id}')

                # update canvas
                point1, = ax1.plot(c.obj1.x, c.obj1.y, 'ro', markersize = 3)
                point2, = ax2.plot(c.obj2.x, c.obj2.y, 'ro', markersize = 3)
                text1 = ax1.text(c.obj1.x, c.obj1.y, f'{id}', fontsize=10, ha='right')
                text2 = ax2.text(c.obj2.x, c.obj2.y, f'{id}', fontsize=10, ha='right')

                # store points and text obj
                plt_ref["points1"][id] = point1
                plt_ref["points2"][id] = point2
                plt_ref["texts1"][id] = text1
                plt_ref["texts2"][id] = text2

            canvas1.draw()
            canvas2.draw()


    def selectImageGUI(self,idx,ax, canvas):
        filename = askopenfilename()
        if(len(filename)!=0):
            self.loadImage(filename,idx)
            self.displayImage(idx,ax, canvas)

    def exit(self,window):
        window.quit()
        window.destroy()
        print("exit program")



    def gui(self):
        window = Tk()
        window.title('Annotation Tool V2')
        window.geometry("1500x700")
        window.protocol("WM_DELETE_WINDOW", lambda: self.exit(window))

        # store points on images and text label ref so when user update the correspondence pair, it can relfect on the images
        plt_ref = {"points1":{},
                   "texts1":{},
                   "points2":{},
                   "texts2":{}}

        # menu bar
        menubar = Menu(window)

        # message display
        msg = Label(window, height= 1, text="Message:", font=("Arial", 12))
        msg.grid(row=3, column=0)

        # image display area
        fig1, ax1 = plt.subplots(figsize=(6, 4 ))
        fig2, ax2 = plt.subplots(figsize=(6, 4 ))
        ax1.set_title("Image 1")
        ax2.set_title("Image 2")
        ax1.axis("off")
        ax2.axis("off")
        canvas1 = FigureCanvasTkAgg(fig1, master=window)
        canvas1.get_tk_widget().grid(row=0,column=0, columnspan=3, padx = 10, pady = 20)
        canvas2 = FigureCanvasTkAgg(fig2, master=window)
        canvas2.get_tk_widget().grid(row=0,column=3, columnspan=3, padx = 10, pady = 20)
        
        default_width = 12
        col_func = 6

        # choose image
        btn_im1 = Button(window, text="Choose image 1", width = default_width, command = lambda: self.selectImageGUI(1,ax1, canvas1))
        btn_im2 = Button(window, text="Choose image 2", width = default_width, command = lambda: self.selectImageGUI(2, ax2, canvas2))
        btn_im1.grid(row=1,column=1,pady = 2)
        btn_im2.grid(row=1,column=4,pady = 2)

        # correspondence pair table
        listbox = Listbox(window, height = 20, 
                  width = 20, 
                  activestyle = 'dotbox')
        listbox.grid(row=0,column=col_func)

        # add correspondence  
        btn_pair = Button(window, text="Add pair", width = default_width, command = lambda: self.addCorrespondenceGUI(ax1,ax2,canvas1,canvas2,listbox,plt_ref, msg))
        btn_pair.grid(row=1,column=col_func,pady = 2)
        
        # remove listitem
        btn_remove = Button(window, text="Remove pair", width = default_width, command= lambda: self.removeCorrespondenceGUI(listbox,canvas1, canvas2, plt_ref))
        btn_remove.grid(row=2,column=col_func,pady = 2)



        # save/load file
        btn_save = Button(window, text="Save file", width = default_width, command = lambda: self.saveInternalDataGUI())
        btn_save.grid(row=3,column=col_func)
        btn_load = Button(window, text="Load file", width = default_width, command = lambda: self.loadInternalDataGUI(listbox, ax1, ax2, canvas1, canvas2, plt_ref))
        btn_load.grid(row=4,column=col_func)

        # output json
        btn_json = Button(window, text="Output json", width = default_width, command = lambda: self.outputTXTGUI())
        btn_json.grid(row=5,column=col_func)

        # quit button
        btn_exit = Button(window, text="Exit", width = default_width,  command = lambda: self.exit(window))
        btn_exit.grid(row=6,column=col_func)

        window.mainloop()


    
if __name__=="__main__":
    tool = AnnotationTool()
    tool.gui()

