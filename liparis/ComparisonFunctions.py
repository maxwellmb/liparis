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
    
    def getRings(self, image, numRings = 10):
        imMax = np.unravel_index(image.argmax(), image.shape)
    
        X,Y = np.indices(image.shape)
        Xcent = X - imMax[0]
        Ycent = Y - imMax[1]
    
        radius = np.sqrt(Xcent**2 + Ycent**2)
        radMax = np.amax(radius)
        divSize = radMax/numRings
    
        ringVals = {}
    
        for i in range(numRings):
            mask = (divSize*i <= radius) & (radius <= divSize*(i+1))
            ringVals[i] = image[mask]
        yPlot = []
        yErr = []
        for i in range(numRings):
            yPlot.append(np.mean(ringVals[i]))
            yErr.append(np.std(ringVals[i]))
    
        return yPlot, yErr  

    def compareFuncs(self, speckle = True, rings = True, selFrac = 0.1):
        """
        
        Displays result of all lucky imaging algorithms for visual comparison
        As well as the rows containing the brightest pixel value, for more 
        quantitative analysis
        
        speckle - Boolean determines which centering method to use (True uses getSpeckleShift, 
        False uses getShift)
        selFrac - Selection fraction, value between 0 and 1
        
        """
        pseRes = self.datacube.pse()
        pseRes = np.abs(pseRes - np.median(pseRes))
        pseRoCo = self.getMaxRowCol(pseRes)
        pseRow = pseRoCo[0]
        pseCol = pseRoCo[1]
        
        classicRes = self.datacube.classic()
        classicRes = np.abs(classicRes - np.median(classicRes))
        classicRoCo = self.getMaxRowCol(classicRes)
        classicRow = classicRoCo[0]
        classicCol = classicRoCo[1]
        
        hybridRes = self.datacube.hybridLI()
        hybridRes = np.abs(hybridRes - np.median(hybridRes))
        hybridRoCo = self.getMaxRowCol(hybridRes)
        hybridRow = hybridRoCo[0]
        hybridCol = hybridRoCo[1]
        
        isfasRes = self.datacube.isfas()
        isfasRes = np.abs(isfasRes - np.median(isfasRes))
        isfasRoCo = self.getMaxRowCol(isfasRes)
        isfasRow = isfasRoCo[0]
        isfasCol = isfasRoCo[1]
        
        saaRes = self.datacube.shiftAndAdd()
        saaRes = np.abs(saaRes - np.median(saaRes))
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
        
        vMin = 1
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
        
        
        plt.figure(figsize = (18,4))
        
        X = np.arange(0,classicRes.shape[0])
        plt.subplot(1,3,1)
        if not rings:
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
        else:
            numRings = 10
            classicPlot, classicErr = self.getRings(classicRes, numRings)
            psePlot, pseErr = self.getRings(pseRes, numRings)
            isfasPlot, isfasErr = self.getRings(isfasRes, numRings)
            hybridPlot, hybridErr = self.getRings(hybridRes, numRings)

            classicPlotNorm = classicPlot / np.amax(classicPlot)
            psePlotNorm = psePlot / np.amax(psePlot)
            isfasPlotNorm = isfasPlot / np.amax(isfasPlot)
            hybridPlotNorm = hybridPlot / np.amax(hybridPlot)

            classicErrNorm = classicErr / np.amax(classicErr)
            pseErrNorm = pseErr / np.amax(pseErr)
            isfasErrNorm = isfasErr / np.amax(isfasErr)
            hybridErrNorm = hybridErr / np.amax(hybridErr)

            classicErrNorm2 = classicErr / np.amax(classicPlot)
            pseErrNorm2 = pseErr / np.amax(psePlot)
            isfasErrNorm2 = isfasErr / np.amax(isfasPlot)
        
            xPlot = np.arange(numRings)
            plt.title('Average Brightness and Standard Deviation in Succesive Annuli')
            plt.xlabel('Annulus Number')
            plt.ylabel('Value')
            plt.semilogy()
            markers, caps, bars = plt.errorbar(xPlot, classicPlotNorm, classicErrNorm, fmt = 'o',color = 'red', 
                    ecolor = 'black', elinewidth = 2, capsize=0, label = 'Classic LI')
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            markers, caps, bars = plt.errorbar(xPlot, psePlotNorm, pseErrNorm, fmt = 'o',color = 'blue', 
                    ecolor = 'brown', elinewidth = 2, capsize=0, label = 'PSE LI')
           
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            
            markers, caps, bars = plt.errorbar(xPlot, isfasPlotNorm, isfasErrNorm, fmt = 'o',color = 'orange', 
                    ecolor = 'green', elinewidth = 2, capsize=0, label = 'Fourier LI')
            
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

            markers, caps, bars = plt.errorbar(xPlot, hybridPlotNorm, hybridErrNorm, fmt = 'o',color = 'gray', 
                    ecolor = 'blue', elinewidth = 2, capsize=0, label = 'Hybrid LI')
            
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

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
    
    def selFracComp(self):
        X = [1]
        for i in range(10):
            X.append(i*10+10)
        
        classicBright = []
        isfasBright = []
        pseBright = []
        
        for x in X:
            classicBright.append(np.amax(self.datacube.classic(selFrac = x/100.0)))
            isfasBright.append(np.amax(self.datacube.isfas(selFrac = x/100.0)))
            pseBright.append(np.amax(self.datacube.pse(selFrac = x/100.0)))
            print(x)
        
        plt.figure()
        plt.title('Peak Brightness vs Selection Percent')
        plt.xlabel('Selection Percent (%)')
        plt.ylabel('Peak Brightness (value)')
        plt.scatter(X, classicBright, color = 'red', label = 'Classic LI')
        plt.scatter(X, isfasBright, color = 'orange', label = 'Fourier LI')
        plt.scatter(X, pseBright, color = 'blue', label = 'PSE LI')
        plt.legend()
        plt.show()

comp = compare(filename = '/Users/grossman/Desktop/tempFiles/orkid_2301160048_0006.fits')
comp.compareFuncs()
