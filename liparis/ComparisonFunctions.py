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
from matplotlib.colors import SymLogNorm
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
    
    def getRings(self, image, numRings = 10, fillIm = False, size = 2):
        imMax = np.unravel_index(image.argmax(), image.shape)
    
        X,Y = np.indices(image.shape)
        Xcent = X - imMax[0]
        Ycent = Y - imMax[1]
    
        radius = np.sqrt(Xcent**2 + Ycent**2)
        if fillIm:
            radMax = np.amax(radius)
            divSize = radMax/numRings
        else:
            divSize = size
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

    def compareFuncs(self, speckle = True, rings = True, selFrac = 0.1, fillIm = False, numRings = 10):
        """
        
        Displays result of all lucky imaging algorithms for visual comparison
        As well as the rows containing the brightest pixel value, for more 
        quantitative analysis
        
        speckle - Boolean determines which centering method to use (True uses getSpeckleShift, 
        False uses getShift)
        selFrac - Selection fraction, value between 0 and 1
        
        """
        pseRes = self.datacube.pse()
        pseRes = pseRes - np.median(pseRes)
        pseRoCo = self.getMaxRowCol(pseRes)
        pseRow = pseRoCo[0]
        pseCol = pseRoCo[1]
        
        classicRes = self.datacube.classic()
        classicRes = classicRes - np.median(classicRes)
        classicRoCo = self.getMaxRowCol(classicRes)
        classicRow = classicRoCo[0]
        classicCol = classicRoCo[1]
        
        hybridRes = self.datacube.hybridLI()
        hybridRes = hybridRes - np.median(hybridRes)
        hybridRoCo = self.getMaxRowCol(hybridRes)
        hybridRow = hybridRoCo[0]
        hybridCol = hybridRoCo[1]
        
        isfasRes = self.datacube.isfas()
        isfasRes = isfasRes - np.median(isfasRes)
        isfasRoCo = self.getMaxRowCol(isfasRes)
        isfasRow = isfasRoCo[0]
        isfasCol = isfasRoCo[1]
        
        saaRes = self.datacube.shiftAndAdd()
        saaRes = saaRes - np.median(saaRes)
        saaRoCo = self.getMaxRowCol(saaRes)
        saaRow = saaRoCo[0]
        saaCol = saaRoCo[1]

        meanRes = self.datacube.mean()
        meanRes = meanRes - np.median(meanRes)
        meanRoCo = self.getMaxRowCol(meanRes)
        meanRow = meanRoCo[0]
        meanCol = meanRoCo[1]
        
        pseMax = np.amax(pseRes)
        classicMax = np.amax(classicRes)
        isfasMax = np.amax(isfasRes)
        saaMax = np.amax(saaRes)
        hybridMax = np.amax(hybridRes)
        meanMax = np.amax(meanRes)
        maxes = [pseMax,classicMax, isfasMax, saaMax, hybridMax, meanMax]
        
        pseMin = np.amin(pseRes)
        classicMin = np.amin(classicRes)
        isfasMin = np.amin(isfasRes)
        saaMin = np.amin(saaRes)
        hybridMin = np.amin(hybridRes)
        meanMin = np.amin(meanRes)
        mins = [pseMin,classicMin, isfasMin, saaMin, hybridMin, meanMin]
        
        vMin = np.amin(mins)
        vMax = np.amax(maxes)
        
        plt.figure(figsize = (12,12))
        if speckle:
            plt.suptitle('Final Images for LI Algorithms, Sp. Shift')
        else:
            plt.suptitle('Final Images for LI Algorithms, CC Shift')
            
        plt.subplot(2,3,1)
        plt.title('Classic Lucky Imaging')
        plt.imshow(classicRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()
        
        plt.subplot(2,3,2)
        plt.title('Fourier Lucky Imaging')
        plt.imshow(isfasRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()
        
        plt.subplot(2,3,3)
        plt.title('Power Spectrum Extended')
        plt.imshow(pseRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()

        
        plt.subplot(2,3,4)
        plt.title('Shift and Add')
        plt.imshow(saaRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()

        plt.subplot(2,3,5)
        plt.title('Hybrid')
        plt.imshow(hybridRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()

        plt.subplot(2,3,6)
        plt.title('Mean')
        plt.imshow(meanRes, cmap = 'gray', vmin = vMin, vmax = vMax)
        plt.colorbar()
    

        plt.savefig('/Users/grossman/Desktop/Plots New/Lin_Norm.png')


        plt.figure(figsize = (12,12))
        if speckle:
            plt.suptitle('Final Images for LI Algorithms, Log Norm, Sp. Shift')
        else:
            plt.suptitle('Final Images for LI Algorithms, Log Norm, CC Shift')
        
          
        sln = SymLogNorm(linthresh = 1, vmin = vMin, vmax = vMax)
        
        plt.subplot(2,3,1)
        plt.title('Classic Lucky Imaging')
        plt.imshow(classicRes, cmap = 'gray', norm = sln)
        plt.colorbar()
        
        plt.subplot(2,3,2)
        plt.title('Fourier Lucky Imaging')
        plt.imshow(isfasRes, cmap = 'gray', norm = sln)
        plt.colorbar()
        
        plt.subplot(2,3,3)
        plt.title('Power Spectrum Extended')
        plt.imshow(pseRes, cmap = 'gray', norm = sln)
        plt.colorbar()
        
        plt.subplot(2,3,4)
        plt.title('Shift and Add')
        plt.imshow(saaRes, cmap = 'gray', norm = sln)
        plt.colorbar()

        plt.subplot(2,3,5)
        plt.title('Hybrid')
        plt.imshow(hybridRes, cmap = 'gray', norm = sln)
        plt.colorbar()

        plt.subplot(2,3,6)
        plt.title('Mean')
        plt.imshow(meanRes, cmap = 'gray', norm = sln)
        plt.colorbar()

        
        plt.savefig('/Users/grossman/Desktop/Plots New/Log_Norm.png')
        
        
        plt.figure()
        if not rings:
            X = np.arange(0,classicRes.shape[0])
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
            classicPlot, classicErr = self.getRings(classicRes)
            psePlot, pseErr = self.getRings(pseRes)
            isfasPlot, isfasErr = self.getRings(isfasRes)
            hybridPlot, hybridErr = self.getRings(hybridRes)
            saaPlot, saaErr = self.getRings(saaRes)
            meanPlot, meanErr = self.getRings(meanRes)

            classicPlotNorm = classicPlot / np.amax(classicPlot)
            psePlotNorm = psePlot / np.amax(psePlot)
            isfasPlotNorm = isfasPlot / np.amax(isfasPlot)
            hybridPlotNorm = hybridPlot / np.amax(hybridPlot)
            saaPlotNorm = saaPlot / np.amax(saaPlot)
            meanPlotNorm = meanPlot / np.amax(meanPlot)

            classicErrNorm = classicErr / np.amax(classicErr)
            pseErrNorm = pseErr / np.amax(pseErr)
            isfasErrNorm = isfasErr / np.amax(isfasErr)
            hybridErrNorm = hybridErr / np.amax(hybridErr)
            saaErrNorm = saaErr / np.amax(saaErr)
            meanErrNorm = meanErr / np.amax(meanErr)

            classicErrNorm2 = classicErr / np.amax(classicPlot)
            pseErrNorm2 = pseErr / np.amax(psePlot)
            isfasErrNorm2 = isfasErr / np.amax(isfasPlot)
        
            xPlot = np.arange(numRings)*2

            plt.title('Average Brightness and Standard Deviation in Succesive Annuli')
            plt.xlabel('Pixels from Center')
            plt.ylabel('Value')
            plt.semilogy()
            markers, caps, bars = plt.errorbar(xPlot, classicPlotNorm, classicErrNorm, fmt = 'o',color = 'red', 
                    ecolor = 'red', elinewidth = 2, capsize=5, label = 'Classic LI')
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

            markers, caps, bars = plt.errorbar(xPlot, psePlotNorm, pseErrNorm, fmt = 'o',color = 'blue', 
                    ecolor = 'blue', elinewidth = 2, capsize=5, marker = 'P', label = 'PSE LI')
           
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            
            markers, caps, bars = plt.errorbar(xPlot, isfasPlotNorm, isfasErrNorm, fmt = 'o',color = 'orange', 
                    ecolor = 'orange', elinewidth = 2, capsize=5, marker = 'h', label = 'Fourier LI')
            
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

            markers, caps, bars = plt.errorbar(xPlot, hybridPlotNorm, hybridErrNorm, fmt = 'o',color = 'gray', 
                    ecolor = 'gray', elinewidth = 2, capsize=5, marker = 's', label = 'Hybrid LI')
            
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

            markers, caps, bars = plt.errorbar(xPlot, saaPlotNorm, saaErrNorm, fmt = 'o',color = 'green', 
                    ecolor = 'green', elinewidth = 2, capsize=5, marker = 'X', label = 'Hybrid LI')
            
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

            markers, caps, bars = plt.errorbar(xPlot, meanPlotNorm, meanErrNorm, fmt = 'o',color = 'yellow', 
                    ecolor = 'yellow', elinewidth = 2, capsize=5, marker = 'D', label = 'Hybrid LI')
            
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

            plt.plot(xPlot, classicPlotNorm, color = 'red')
            plt.plot(xPlot, psePlotNorm, color = 'blue')
            plt.plot(xPlot, isfasPlotNorm, color = 'orange')
            plt.plot(xPlot, hybridPlotNorm, color = 'gray')
            plt.plot(xPlot, saaPlotNorm, color = 'green')
            plt.plot(xPlot, meanPlotNorm, color = 'yellow')

            plt.legend()

        plt.savefig('/Users/grossman/Desktop/Plots New/Rings.png')
    
        plt.show()
        
        return
    
    def selFracComp(self):
        X = [1]
        for i in range(10):
            X.append(i*10+10)
        
        classicBright = []
        #isfasBright = []
        pseBright = []
        
        for x in X:
            classicBright.append(np.amax(self.datacube.classic(selFrac = x/100.0)))
            #isfasBright.append(np.amax(self.datacube.isfas(selFrac = x/100.0)))
            pseBright.append(np.amax(self.datacube.pse(selFrac = x/100.0)))
            print(x)
        
        plt.figure()
        plt.title('Peak Brightness vs Selection Percent')
        plt.xlabel('Selection Percent (%)')
        plt.ylabel('Peak Brightness (value)')
        plt.scatter(X, classicBright, color = 'red', label = 'Classic LI')
        #plt.scatter(X, isfasBright, color = 'orange', label = 'Fourier LI')
        plt.scatter(X, pseBright, color = 'blue', label = 'PSE LI')
        plt.legend()
        
        plt.savefig('/Users/grossman/Desktop/Plots New/selFracComp')

        plt.show()

comp = compare(filename = '/Users/grossman/Desktop/tempFiles/orkid_2301160048_0006.fits')
comp.compareFuncs()
