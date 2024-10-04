import cv2 as cv
#import matplotlib.pyplot.ginput as ginput
import pickle

from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter.filedialog import askopenfilename, asksaveasfilename


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

class AnnotationTool:
    def __init__(self) -> None:
        # images
        self.im1 = None
        self.im2 = None
        
        # correspondence pair data
        self.pair={}

        # some GUI function "mutex"
        self.inuse_removeCorrespondenceGUI = False
        self.inuse_addCorrespondenceGUI = False

    # add/remove of images and correspondence 
    def setImage(self, image_path, idx:int):
        im = mpimg.imread(image_path)
        if idx == 1:         
            self.im1 = im
        elif idx ==2:
            self.im2 = im
        else:
            print("setImage: image index must less than 2!")
        
        if(self.im1 is not None and self.im2 is not None):
            if(self.im1.shape != self.im2.shape):
                print("Warning: two images size are not the same.")
    
    def resetImage(self, idx:int):
        if idx == 1:
            self.im1 = None
        elif idx == 2:
            self.im2 = None
        else:
            print("resetImage: image index must less than 2!")

    def addCorrespondence(self,c:Correspondence):
        self.pair[c.id] = c

    def removeCorrespondence(self,id:int):
        if id in self.pair:
            self.pair.pop(id)
        else:
            print(f"id {id} does not exist.")

    def resetCorrespondence(self):
        self.pair.clear()

    def reset(self):
        self.resetImage(1)
        self.resetImage(2)
        self.resetCorrespondence()

    def saveCorrespondence(self,filename:str = None):
        if len(self.pair)==0:
            print("No data to save.")
            return
        
        if filename is None:
            filename = "correspondence.pkl"

        print(f"save correspondence pair to {filename}")
        with open(filename,'wb') as file:
            pickle.dump(self.pair, file)
            print("Correspondence pair saved!")
            file.close()

    def loadCorrespondence(self,filepath:str):
        print(f"load correspondence pari from {filepath}")
        with open(filepath,'rb') as file:
            self.pair = pickle.load(file)
            print("Correspondence pair loaded!")
            file.close()

        # need to update Correspondence id
        keys = self.pair.keys()
        Correspondence.id = max(keys)


    ######################################## use for GUI #############################################

    def displayImage(self, idx, ax, canvas):
        if(idx==1):
            im = self.im1
        elif (idx==2):
            im = self.im2
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
        if(self.im1 is None or self.im2 is None):
            print("Not enought images.")
            msg.config(text="Message: Not enought images.")
            self.inuse = False
            return
        # point 1 from image 1
        plt.sca(ax1)
        msg.config(text="Message: Pick one point on image 1.")
        x1, y1 = plt.ginput(1)[0]
        point1, = ax1.plot(x1, y1, 'ro', markersize = 3)
        canvas1.draw()
        msg.config(text="Message: Pick one point on image 2.")
        # point 2 from image 2
        plt.sca(ax2)
        x2, y2 = plt.ginput(1)[0]
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



    def saveCorrespondenceGUI(self):
        filename = asksaveasfilename(defaultextension=".pkl",filetypes=(("pickle file", "*.pkl"),("All Files", "*.*")))
        self.saveCorrespondence(filename)

    def loadCorrespondenceGUI(self):
        filename = askopenfilename()
        self.loadCorrespondence(filename)

    def selectImageGUI(self,idx,ax, canvas):
        filename = askopenfilename()
        self.setImage(filename,idx)
        self.displayImage(idx,ax, canvas)

    def exit(self,window):
        window.quit()
        window.destroy()
        print("exit program")



    def gui(self):
        window = Tk()
        window.title('Annotation Tool')
        window.geometry("1020x650")
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
        fig1, ax1 = plt.subplots(figsize=(4, 4 ))
        fig2, ax2 = plt.subplots(figsize=(4, 4 ))
        ax1.set_title("Image 1")
        ax2.set_title("Image 2")
        ax1.axis("off")
        ax2.axis("off")
        canvas1 = FigureCanvasTkAgg(fig1, master=window)
        canvas1.get_tk_widget().grid(row=0,column=0, padx = 10, pady = 20)
        canvas2 = FigureCanvasTkAgg(fig2, master=window)
        canvas2.get_tk_widget().grid(row=0,column=1, padx = 10, pady = 20)
        

        # correspondence pair table
        listbox = Listbox(window, height = 20, 
                  width = 20, 
                  activestyle = 'dotbox')
        listbox.grid(row=0,column=2)

        default_width = 12


        # choose image
        btn_im1 = Button(window, text="Choose image 1", width = default_width, command = lambda: self.selectImageGUI(1,ax1, canvas1))
        btn_im2 = Button(window, text="Choose image 2", width = default_width, command = lambda: self.selectImageGUI(2, ax2, canvas2))
        btn_im1.grid(row=1,column=0,pady = 2)
        btn_im2.grid(row=1,column=1,pady = 2)


        # add correspondence  
        btn_pair = Button(window, text="Add pair", width = default_width, command = lambda: self.addCorrespondenceGUI(ax1,ax2,canvas1,canvas2,listbox,plt_ref, msg))
        btn_pair.grid(row=1,column=2,pady = 2)
        
        # remove listitem
        btn_remove = Button(window, text="Remove pair", width = default_width, command= lambda: self.removeCorrespondenceGUI(listbox,canvas1, canvas2, plt_ref))
        btn_remove.grid(row=2,column=2,pady = 2)



        # save/load file
        btn_save = Button(window, text="Save file", width = default_width, command = lambda: self.saveCorrespondenceGUI())
        btn_save.grid(row=3,column=2)
        btn_load = Button(window, text="Load file", width = default_width, command = lambda: self.loadCorrespondenceGUI())
        btn_load.grid(row=4,column=2)

        # quit button
        btn_exit = Button(window, text="Exit", width = default_width,  command = lambda: self.exit(window))
        btn_exit.grid(row=5,column=2)

        window.mainloop()


    
if __name__=="__main__":
    tool = AnnotationTool()
    tool.gui()

