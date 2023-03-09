from astropy.io import fits
import numpy as np
import skimage.io
import skimage.filters
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import Datacube
class compare():

    def __init__(self, filename):
        self.file = filename
        self.datacube = Datacube.Datacube(self.file, writeTo = False)

    def getMaxRowCol(self, im):
        """
        
        Returns row and column containing brighest pixel in im
        
        im - Image to get row and column from
        
        """
        maxInd = np.unravel_index(im.argmax(), im.shape)
        row = im[maxInd[0]]
        col = im.T[maxInd[1]]
        return row, col

    def compareFuncs(self, speckle = True, selFrac = 0.1):
        """
        
        Displays result of all lucky imaging algorithms for visual comparison
        As well as the rows containing the brightest pixel value, for more 
        quantitative analysis
        
        speckle - Boolean determines which centering method to use (True uses getSpeckleShift, 
        False uses getShift)
        selFrac - Selection fraction, value between 0 and 1
        
        """
        pseRes = self.datacube.pse()
        pseRoCo = self.getMaxRowCol(pseRes)
        pseRow = pseRoCo[0]
        pseCol = pseRoCo[1]
        
        classicRes = self.datacube.classic()
        classicRoCo = self.getMaxRowCol(classicRes)
        classicRow = classicRoCo[0]
        classicCol = classicRoCo[1]
        
        hybridRes = self.datacube.hybridLI()
        hybridRoCo = self.getMaxRowCol(hybridRes)
        hybridRow = hybridRoCo[0]
        hybridCol = hybridRoCo[1]
        
        isfasRes = self.datacube.isfas()
        isfasRoCo = self.getMaxRowCol(isfasRes)
        isfasRow = isfasRoCo[0]
        isfasCol = isfasRoCo[1]
        
        saaRes = self.datacube.shiftAndAdd()
        saaRoCo = self.getMaxRowCol(saaRes)
        saaRow = saaRoCo[0]
        saaCol = saaRoCo[1]
        
        pseMax = np.amax(pseRes)
        classicMax = np.amax(classicRes)
        isfasMax = np.amax(isfasRes)
        saaMax = np.amax(saaRes)
        hybridMax = np.amax(hybridRes)
        maxes = [pseMax,classicMax, isfasMax, saaMax, hybridMax]
        
        pseMin = np.amin(pseRes)
        classicMin = np.amin(classicRes)
        isfasMin = np.amin(isfasRes)
        saaMin = np.amin(saaRes)
        hybridMin = np.amin(hybridRes)
        mins = [pseMin,classicMin, isfasMin, saaMin, hybridMin]
        
        vMin = np.amin(mins)
        vMax = np.amax(maxes)
        
        plt.figure(figsize = (9,9))
        if speckle:
            plt.suptitle('Final Images for LI Algorithms, Sp. Shift')
        else:
            plt.suptitle('Final Images for LI Algorithms, CC Shift')
            
        plt.subplot(2,2,1)
        plt.title('Classic Lucky Imaging')
        plt.imshow(classicRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()
        
        plt.subplot(2,2,2)
        plt.title('Fourier Lucky Imaging')
        plt.imshow(isfasRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()
        
        plt.subplot(2,2,3)
        plt.title('Power Spectrum Extended')
        plt.imshow(pseRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()

        
        plt.subplot(2,2,4)
        plt.title('Shift and Add')
        plt.imshow(saaRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()
        
        plt.show()
        
        plt.figure(figsize = (9,9))
        if speckle:
            plt.suptitle('Final Images for LI Algorithms, Log Norm, Sp. Shift')
        else:
            plt.suptitle('Final Images for LI Algorithms, Log Norm, CC Shift')
        
          
        ln = LogNorm(vmin = vMin, vmax = vMax)
        plt.subplot(2,2,1)
        plt.title('Classic Lucky Imaging')
        plt.imshow(classicRes, cmap = 'gray', norm = ln)
        plt.colorbar()
        
        plt.subplot(2,2,2)
        plt.title('Fourier Lucky Imaging')
        plt.imshow(isfasRes, cmap = 'gray', norm = ln)
        plt.colorbar()
        
        plt.subplot(2,2,3)
        plt.title('Power Spectrum Extended')
        plt.imshow(pseRes, cmap = 'gray', norm = ln)
        plt.colorbar()
        
        plt.subplot(2,2,4)
        plt.title('Shift and Add')
        plt.imshow(saaRes, cmap = 'gray', norm = ln)
        plt.colorbar()
        
        plt.show()
        
        plt.figure(figsize = (18,4))
        
        X = np.arange(0,classicRes.shape[0])
        plt.subplot(1,3,1)
        plt.title('Rows and Columns Containing Maximum')
        plt.xlabel('Coordinate')
        plt.ylabel('Value')
        plt.plot(X, classicRow, label = 'Classic LI Row', color = 'red')
        #plt.plot(X, classicCol, label = 'Classic LI Col', color = 'red', linestyle = 'dotted')
        plt.plot(X, isfasRow, label = 'Fourier LI Row', color = 'green')
        #plt.plot(X, isfasCol, label = 'Fourier LI Col', color = 'green', linestyle = 'dotted')
        plt.plot(X, pseRow, label = 'PSE LI Row', color = 'blue')
        #plt.plot(X, pseCol, label = 'PSE LI Col', color = 'blue', linestyle = 'dotted')
        plt.plot(X, hybridRow, label = 'Hybrid LI Row', color = 'black')
        #plt.plot(X, classicRevCol, label = 'RC LI Col', color = 'black', linestyle = 'dotted')
        plt.plot(X, saaRow, label = 'SAA Row', color = 'orange')
        #plt.plot(X, classicRevCol, label = 'RC LI Col', color = 'black', linestyle = 'dotted')
        plt.legend()
        
        plt.subplot(1,3,2)
        plt.title('Hybrid Lucky Imaging')
        plt.imshow(hybridRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()
        
        plt.subplot(1,3,3)
        plt.title('Hybrid Lucky Imaging')
        plt.imshow(hybridRes, cmap = 'gray', norm = ln)
        plt.colorbar()
        
        plt.show()
        
        return

comp = compare(filename = '/Users/grossman/Desktop/tempFiles/orkid_2301160048_0006.fits')
comp.compareFuncs()
print('e')