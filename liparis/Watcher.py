from watchdog.events import FileSystemEventHandler
from watchdog.events import PatternMatchingEventHandler
#import matplotlib
#matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import Datacube

class HandleNewFile(FileSystemEventHandler):

    def __init__(self,output_path,fig,axes,xCent,xSize,yCent,ySize):
        self.xCent = xCent
        self.xSize = xSize
        self.yCent = yCent
        self.ySize = ySize
        self.output_path = output_path        
        self.fig = fig
        self.axes = axes
        #This is inelegant, but a fix for now
        self.cbar00 = None
        self.cbar01 = None
        self.cbar10 = None
        self.cbar11 = None
        self.cbar20 = None
        self.cbar21 = None
        self.cbar30 = None
        self.cbar31 = None
       

    def on_created(self, event):
        time.sleep(5)
        print('\n Found new file: ' + event.src_path)
        try: 
            cube = Datacube.Datacube(event.src_path, xCent = self.xCent, xSize = self.xSize, yCent = self.yCent, ySize = self.ySize)
        except TypeError:
            print("The file might still be writing, let's wait 10s before trying again")
            time.sleep(30)
            cube = Datacube.Datacube(event.src_path, xCent = self.xCent, xSize = self.xSize, yCent = self.yCent, ySize = self.ySize)
        print('\n Found new file: ' + event.src_path)
        #self.fig.clf()
        self.fig.suptitle(event.src_path)
        # plt.title(event.src_path)

        print('Running shift and add algorithm.')
        shift_add = cube.shiftAndAdd(outDir=self.output_path)
        print("Done shift and add")
        im00 = self.axes[0].imshow(shift_add,origin='lower')
        if self.cbar00 is None:
            self.cbar00 = plt.colorbar(im00,ax=self.axes[0])
        else:
            self.cbar00.update_normal(im00)
        im01 = self.axes[4].imshow(shift_add,norm=LogNorm(),origin='lower')
        if self.cbar01 is None:
            self.cbar01 = plt.colorbar(im01,ax=self.axes[4])
        else:
            self.cbar01.update_normal(im01)
        
        ### Classic Lucky Imaging
        print('Running classic lucky imaging algorithm.')
        classic_im = cube.classic(outDir = self.output_path)
        print('Done with classic algorithm on file '+ event.src_path+'.\n')
        im10 = self.axes[1].imshow(classic_im,origin='lower')
        if self.cbar10 is None:
            self.cbar10 = plt.colorbar(im10,ax=self.axes[1])
        else:
            self.cbar10.update_normal(im10)
        im11 = self.axes[5].imshow(classic_im,norm=LogNorm(),origin='lower')
        if self.cbar11 is None:
            self.cbar11 = plt.colorbar(im11,ax=self.axes[5])
        else:
            self.cbar11.update_normal(im11)
        # plt.show()
        
        ### PSE Method
        print('Running Power Spectrum Extended algorithm.')
        pse_im = cube.pse(outDir = self.output_path)
        print('Done with Power Spectrum Extended algorithm on file '+ event.src_path+'.\n')
        im20=self.axes[2].imshow(pse_im,origin='lower')
        if self.cbar20 is None:
            self.cbar20 = plt.colorbar(im20,ax=self.axes[2])
        else:
            self.cbar20.update_normal(im20)
        im21 = self.axes[6].imshow(pse_im,norm=LogNorm(),origin='lower')
        if self.cbar21 is None:
            self.cbar21 = plt.colorbar(im21,ax=self.axes[6])
        else:
            self.cbar21.update_normal(im21)  
        print('Running Stack algorithm.')
        stack_im = cube.stack(outDir = self.output_path)
        print('Done with Stack algorithm on file '+ event.src_path + '.\n')
        im30 = self.axes[3].imshow(stack_im, origin = 'lower')
        if self.cbar30 is None:
            self.cbar30 = plt.colorbar(im30, ax = self.axes[3])
        else:
            self.cbar30.update_normal(im30)
        im31 = self.axes[7].imshow(stack_im, norm = LogNorm(), origin = 'lower')
        if self.cbar31 is None:
            self.cbar31 = plt.colorbar(im31, ax = self.axes[7])
        else:
            self.cbar31.update_normal(im31)
"""
        ### Fourier Lucky Imaging
        print('Running fourier algorithm on file.')
        fourier_im = cube.isfas(outDir = self.output_path)
        print('Done with fourier algorithm on file '+ event.src_path+'.\n')
        im30 = self.axes[3].imshow(fourier_im,origin='lower')
        if self.cbar30 is None:
            self.cbar30 = plt.colorbar(im30,ax=self.axes[3])
        else:
            self.cbar30.update_normal(im30)
        im31 = self.axes[7].imshow(fourier_im,norm=LogNorm(),origin='lower')
        if self.cbar31 is None:
            self.cbar31 = plt.colorbar(im31,ax=self.axes[7])
        else:
            self.cbar31.update_normal(im31)
        
        # plt.show()
        ### Shift and add 
"""                
       

class PatternHandleNewFile(PatternMatchingEventHandler):
#This was originally to solve some issues with watchdog not triggering when other users
#modified files.  Switching Observer to PollingObserver has since solved this.
#As such, this class is not used, but kept here just in case.
    def __init__(self,output_path,fig,axes, matchPatterns = ['*.fits']):
        self.output_path = output_path
        PatternMatchingEventHandler.__init__(self,patterns = matchPatterns, ignore_directories = True, case_sensitive = False)
        self.fig = fig
        self.axes = axes

    def on_created(self, event):
        time.sleep(5)
        print('\n Found new file: ' + event.src_path)
        try: 
            cube = Datacube.Datacube(event.src_path)
        except TypeError:
            print("The file might still be writing, let's wait 10s before trying again")
            time.sleep(10)
            cube = Datacube.Datacube(event.src_path)
        print('\n Found new file: ' + event.src_path)
        self.fig.suptitle(event.src_path)
        # plt.title(event.src_path)

        print('Running shift and add algorithm.')
        shift_add = cube.shiftAndAdd(outDir=self.output_path)
        print("Done shift and add")
        im00 = self.axes[0].imshow(shift_add,origin='lower')
        plt.colorbar(im00,ax=self.axes[0])
        im01 = self.axes[4].imshow(shift_add,norm=LogNorm(),origin='lower')
        plt.colorbar(im01,ax=self.axes[4])
        
        ### Classic Lucky Imaging
        print('Running classic lucky imaging algorithm.')
        classic_im,_ = cube.classic(outDir = self.output_path)
        print('Done with classic algorithm on file '+ event.src_path+'.\n')
        im10 = self.axes[1].imshow(classic_im,origin='lower')
        plt.colorbar(im10,ax=self.axes[1])
        im11 = self.axes[5].imshow(classic_im,norm=LogNorm(),origin='lower')
        plt.colorbar(im11,ax=self.axes[5])
        # plt.show()
        
        ### PSE Method
        print('Running Power Spectrum Extended algorithm.')
        pse_im = cube.pse(outDir = self.output_path)
        print('Done with Power Spectrum Extended algorithm on file '+ event.src_path+'.\n')
        im20=self.axes[2].imshow(pse_im,origin='lower')
        plt.colorbar(im20,ax=self.axes[2])
        im21 = self.axes[6].imshow(pse_im,norm=LogNorm(),origin='lower')
        plt.colorbar(im21,ax=self.axes[6])
"""
        ### Fourier Lucky Imaging
        print('Running fourier algorithm on file.')
        fourier_im = cube.isfas(outDir = self.output_path)
        print('Done with fourier algorithm on file '+ event.src_path+'.\n')
        im30 = self.axes[3].imshow(fourier_im,origin='lower')
        plt.colorbar(im30,ax=self.axes[3])
        im31 = self.axes[7].imshow(fourier_im,norm=LogNorm(),origin='lower')
        plt.colorbar(im31,ax=self.axes[7])
        
        # plt.show()

"""
