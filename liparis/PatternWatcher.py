from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
#import matplotlib
#matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import Datacube

class PatternHandleNewFile(PatternMatchingEventHandler):

    def __init__(self,output_path,patternMatches = ['*.fits'],fig,axes):
        self.output_path = output_path
        PatternMatchingEventHandler.__init__(self,patterns=patternMatches, ignore_directories = True, case_sensitive = False)
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
        ### Shift and add 
        # print('Running shift and add')
"""        
        


