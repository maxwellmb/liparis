from astropy.io import fits
import numpy as np
import skimage.io
import skimage.filters
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from pathlib import PurePosixPath

class Datacube():
    
    def __init__(self, filename, outDir = './', xCent = None, yCent = None, xSize = None, ySize = None, writeTo = True):
        """
        
        Currently configured to work with data from VAMPIRES
        Might need rewriting once I have access to ORKID data
        
        Add header
        Edit history
        add self.pse
        self.classic 
        etc.
        
        """
        self.outDir = outDir
        self.file = filename
        self.writeTo = writeTo
        with fits.open(filename) as hdul:
            self.rawIm1 = hdul[0].data[0]
            self.rawDim0 = self.rawIm1.shape[0]
            self.rawDim1 = self.rawIm1.shape[1] 
            self.header = hdul[0].header
            if xCent is not None:
                xLow = xCent-int(xSize/2)
                xLow = xLow*(xLow>0)
                xHigh = xCent + int(xSize/2)
                xHigh = xHigh*(xHigh <= self.rawDim1) + self.rawDim1*(xHigh > self.rawDim1)
                yLow = yCent - int(ySize/2)
                yLow = yLow*(yLow>0)
                yHigh = yCent + int(ySize/2)
                yHigh = yHigh*(yHigh <= self.rawDim0) + self.rawDim0*(yHigh > self.rawDim0)
                self.imageRaw = hdul[0].data[:,yLow:yHigh,xLow:xHigh].astype(float)
            else:            
                self.imagesRaw = hdul[0].data.astype(float)
            #self.images = hdul[0].data[:,xlow:xhigh,ylow:yhigh]
        #self.images = np.array([hdu.data for hdu in imageData[1:]])

        #self.images = gaussian_filter(self.imagesRaw, sigma = 2)
        self.images = self.imagesRaw
        self.image1 = self.images[0]
        self.imageDim0 = self.image1.shape[0]
        self.imageDim1 = self.image1.shape[1]
        self.numImages = self.images.shape[0]

    def blur(self, image, sigma = 1/np.sqrt(2*np.log(2))):
        """
        
        The isfas algorithm calls for a gaussian blur to be applied
        Returns result in fourier domain
        
        image - Image to blur
        sigma - size of kernel of gaussian blur, taken from isfas paper (see isfas function)
        
        """
        fft = np.fft.fft2(image)
        fftshift = np.fft.fftshift(fft)
        fftabs = np.abs(fftshift)
        #fftabs = np.abs(fft)
        blurFFT = skimage.filters.gaussian(fftabs, sigma, mode = 'reflect', preserve_range = True, truncate = 1)
        return blurFFT
    
    def getShift(self, image, image2 = None, usf = 100):
        """
        
        Uses cross-correlation to compute image offset
        Returns shifted image
        Found that it might not work well for poor (un-selected) data
        image - Image to center
        image2 - Optional alternate reference image
        usf - upsample factor, 100 found to not slow it down too much,
        
        """
        if image2 is not None:
            shift, error, diffphase = phase_cross_correlation(image2 - np.median(image2), image - np.median(image), upsample_factor=usf, normalization = None)
        else:
            shift, error, diffphase = phase_cross_correlation(self.image1 - np.median(self.image1), image - np.median(image), upsample_factor=usf, normalization = None)
        #return shift
        imageShift = np.fft.ifftn(ndimage.fourier_shift(np.fft.fftn(image), shift))
        return imageShift
    
    def fourier_filter(self,rows = [44,132]):
        '''
        In some cases there
        The rows here are in the fourier domain and have only been tested in one case. 
        '''
        for i in range(self.images.shape[0]):

            # def fourier_filter(data,rows = [44,132]):
            data_fft = np.fft.fftshift(np.fft.fft2(self.images[i]))
            
            #Replace these rows with the median absolute value
            for row in rows:
                data_fft[row] = 0.5*(data_fft[row-1]+data_fft[row+1])
                    # np.median(np.abs(data_fft))
            
            # data_fft[:,88] = 0.5*(data_fft[:,87]+data_fft[:,89])
            filtered = np.fft.ifft2(np.fft.ifftshift(data_fft))

            self.images[i] = np.abs(filtered)

    def median_row_subtract(self,n_columns=10):
        '''
        Subtract the mean of each row as measured from the first and last n_columns
        '''
        for i in range(self.images.shape[0]):
            first_last = np.delete(self.images[i],np.s_[n_columns:-n_columns],axis=1)
            first_last_median = np.median(first_last,axis=1)
            self.images[i] = self.images[i]-first_last_median[:,None]

    def getSpeckleShift(self, image, zoom = False, zF = 4.0):
        """
        
        Recenters image based on brightest pixel
        image - Image to center
        zF - zoom factor
        Using zoom slows it down quite a bit
        
        """
        if zoom:
            refImZoom = ndimage.zoom(self.image1, zF)
            imageZoom = ndimage.zoom(image, zF)
            refImZoomBlur = gaussian_filter(refInZoom, sigma = 2)
            imageZoomBlur = gaussian_filter(imageZoom, sigma = 2)
            maxInd1 = np.unravel_index(refImZoomBlur.argmax(), refImZoomBlur.shape)
            maxInd2 = np.unravel_index(imageZoomBlur.argmax(), imageZoomBlur.shape)
            shift = (np.asarray(maxInd1) - np.asarray(maxInd2))/zF
            image2Shift = np.fft.ifftn(ndimage.fourier_shift(np.fft.fftn(image),shift))
        else:
            image1Blur = gaussian_filter(self.image1, sigma = 2)
            image2Blur = gaussian_filter(image, sigma = 2)
            maxInd1 = np.unravel_index(image1Blur.argmax(), image1Blur.shape)
            maxInd2 = np.unravel_index(image2Blur.argmax(), image2Blur.shape)
            shift = (np.asarray(maxInd1) - np.asarray(maxInd2))
            image2Shift = np.fft.ifftn(ndimage.fourier_shift(np.fft.fftn(image),shift))
        return image2Shift
    
    def getHighestValsInd(self, arry, selFrac = 0.1):
        """
        
        Gets the indices of the highest values in 1 dimensional array
        Used in a couple of other functions
        The variable 'arg' is the number of elements to select
        
        arry - Array to get values from
        selFrac - Selection fraction, value between 0 and 1
        
        """
        arg = round(len(arry)*selFrac)
        ind = np.argpartition(arry,-1*arg)[-1*arg:]
        return ind
    
    def brightSelection(self, ims, selFrac = 0.1):
        """
        
        Uses the value of the brightest pixel as selection parameter
        
        ims - Images to select from
        selFrac - Selection fraction, value between 0 and 1
        
        """
        numIms = ims.shape[0]
        brightVals = np.zeros(numIms)
        for i in range(numIms):
            brightVals[i] = np.amax(ims[i])
        selInd = self.getHighestValsInd(brightVals, selFrac)
        return ims[selInd]
    
    def ratioSelection(self, ims, selFrac = 0.1):
        """
        
        Uses the value of the brightest pixel divided by the total
        flux as selection parameter
        
        ims - Images to select from
        selFrac - Selection fraction, value between 0 and 1
        
        """
        numIms = ims.shape[0]
        ratioVals = np.zeros(numIms)
        for i in range(numIms):
            ratioVals[i] = np.amax(ims[i])/np.sum(ims[i])
        selInd = self.getHighestValsInd(ratioVals, selFrac)
        return ims[selInd]
    
    def pse(self,  outDir = None, speckle = True, altCCShift = False, ratio = False, selFrac = 0.1):
        """
        
        Power Spectrum Extended
        Detailed in paper "Post-AO image reconstruction with the PSE algorithm"
        by Cottalorda et. al.
        
        speckle - Boolean determines which centering method to use (True uses getSpeckleShift, 
        False uses getShift)
        
        selFrac - Selection Fraction, value between 0 and 1
        
        """
        hdr = self.header
        
        hdr["ALGO"] = ('Power Spectrum Extended', 'Algorithm used in the creation of this image')
        hdr["SELFRAC"] = (selFrac, 'Portion (between 0 and 1) of data selected')
        
        path = PurePosixPath(self.file)
        fileName = path.stem + '_PSE'
        
        if outDir is None:
            outDir = self.outDir
        
        #print("PSE: " + str(selFrac))
        
        if ratio:
            selIms = self.ratioSelection(self.images, selFrac = selFrac)
        else:
            selIms = self.brightSelection(self.images, selFrac = selFrac)
        numSelIms = len(selIms)
        image1 = selIms[0]
        
        selShiftIms = np.zeros([numSelIms, self.imageDim0, self.imageDim1])
        if speckle:
            for i in range(numSelIms):
                selShiftIms[i] = self.getSpeckleShift(selIms[i])
        else:
            for i in range(numSelIms):
                if not altCCShift:
                    selShiftIms[i] = self.getShift(selIms[i]).real
                else:
                    selfShiftIms[i] = self.getShift(selIms[i], selIms[(i-1)*(i!=0)]).real
        shiftAndAdd = np.sum(selShiftIms, axis = 0)
        shiftAndAddFFT = np.fft.fftshift(np.fft.fftn(shiftAndAdd))
        phase = np.angle(shiftAndAddFFT)
        
        selPSIms = np.zeros([numSelIms, self.imageDim0, self.imageDim1])
        for i in range(numSelIms):
            selPSIms[i] = np.abs(np.fft.fftshift(np.fft.fftn(selIms[i])))**2
        avPS = np.sqrt(np.mean(selPSIms, axis = 0))
        
        recPS = avPS * np.exp(1j*phase)
        
        finalImage = np.abs(np.fft.ifftn(recPS))
        if self.writeTo:
            hdu = fits.ImageHDU(finalImage, hdr)
        
            hdu.writeto(outDir+fileName+'.fits', overwrite = True)
        
            print(outDir+fileName+'.fits')
        
        return finalImage
    
    def isfas(self, outDir = None, speckle = True, altCCShift = True, cutoffDenom = 300.0, selFrac = 0.1, 
                return_ft_avg_only=False):
        """
        
        Image Synthesis by Fourier Amplitude Selection (also called the Fourier method)
        Detailed in paper: "A Highly Efficient Lucky Imaging Algorithm: Image Synthesis
        Based on Fourier Amplitude Selection""
        by Garrel et. al.
        
        speckle - Boolean determines which centering method to use (True uses getSpeckleShift, 
        False uses getShift)
        
        cutoffDenom - The isfas algorithm only considers frequency values 'within a cutoff'
        (no specifics given).  Highest weight value divided by 300 found to be suitable
        
        selFrac - Selection Fraction, value between 0 and 1
        
        """

        ### Update the header
        hdr = self.header
        hdr["ALGO"] = ('Fourier', 'Algorithm used in the creation of this image')
        hdr["SELFRAC"] = (selFrac, 'Portion (between 0 and 1) of data selected')
        
        #print("ISFAS: " + str(selFrac))
        
        if outDir is None:
            outDir = self.outDir
        
        path = PurePosixPath(self.file)
        fileName = outDir+path.stem + '_FOURIER.fits'
        
        imageCubeFFT = np.zeros((self.numImages,self.imageDim0,self.imageDim1), dtype = 'complex_')
        imageCubeMag = np.zeros((self.numImages,self.imageDim0,self.imageDim1))
        for i in range(self.numImages):
            tempImage = self.images[i]
            if not speckle:
                if not altCCShift:
                    selShiftIms[i] = self.getShift(selIms[i]).real
                else:
                    selfShiftIms[i] = self.getShift(selIms[i], selIms[(i-1)*(i!=0)]).real
            else:
                tempImageShift = self.getSpeckleShift(tempImage)
            tempImageFFT = np.fft.fftn(tempImageShift)
            imageCubeFFT[i] = np.fft.fftshift(tempImageFFT) #HERE ------
            tempImageBlur = self.blur(tempImageShift)
            imageCubeMag[i] = tempImageBlur
        
        # avgMod = np.zeros([self.imageDim0,self.imageDim1], dtype = 'complex_')
        # for i in range(self.numImages):
        #     avgMod += imageCubeMag[i]
        # avgMod /= self.numImages
        avgMod = np.mean(imageCubeMag,axis=0)
        avgModMag = np.abs(avgMod)
           
        avgMax = np.amax(avgModMag)

        if return_ft_avg_only: 
            return avgModMag
        cutoff = avgMax/cutoffDenom
        

        y,x = np.indices(avgModMag.shape)
        x = x-avgModMag.shape[1]/2
        y = y-avgModMag.shape[0]/2
        rads = np.sqrt(x**2+y**2)


        indCut = avgModMag < cutoff
        for i in range(self.numImages):
            # imageCubeFFT[i][indCut] = 0
            
            #Testing here. If we're nyquist sampled then this is the cutoff frequency.
            #This should change with wavelength
            imageCubeFFT[i][rads>avgModMag.shape[0]/2] = 0
        
        """
        plt.figure()
        plt.imshow(avgModMag, cmap = 'gray', norm = LogNorm())
        plt.colorbar()
        plt.show()
        

        indCut = avgModMag < cutoff
        for i in range(numImages):
            imageCubeFFT[i][indCut] = 0
        
        maskUsed = np.ones((imageDim0, imageDim1))
        maskUsed[indCut] = 0
        plt.figure(figsize = (18,4))
        plt.suptitle('Average Blurred Modulus and Mask Used')
        plt.subplot(1,3,1)
        plt.imshow(avgModMag, cmap='gray', norm = LogNorm())
        plt.colorbar()
        plt.title('Average Blurred Modulus')
        
        avgModMag[indCut] = 0
        
        plt.subplot(1,3,2)
        plt.title('Mask Used')
        plt.imshow(maskUsed, cmap = 'gray')
        plt.colorbar()
        
        plt.subplot(1,3,3)
        plt.title('Masked Blurred Modulus')
        plt.imshow(avgModMag, cmap = 'gray', norm = LogNorm())
        plt.colorbar()
        
        plt.show()
        """
        
        temparryMag = np.zeros(self.numImages)
        temparryFFT = np.zeros(self.numImages, dtype = 'complex_')
        finalImageFFT = np.zeros((self.imageDim0,self.imageDim1), dtype = 'complex_')
        for i in range(self.imageDim0):
            for j in range(self.imageDim1):
                # for k in range(self.numImages):
                #     temparryMag[k] = imageCubeMag[k][i][j]
                #     temparryFFT[k] = imageCubeFFT[k][i][j]
                #MMB I think this should be faster, no? 
                temparryMag[:] = imageCubeMag[:,i,j]
                temparryFFT[:] = imageCubeFFT[:,i,j]
                ind = self.getHighestValsInd(temparryMag,selFrac)
                finalImageFFT[i][j] = np.mean(temparryFFT[ind])        
        
        
        finalImage =  np.fft.ifftshift(np.abs(np.fft.ifftshift(np.fft.ifftn(finalImageFFT))))
        
        if self.writeTo:
        ## Write your files
            hdu = fits.ImageHDU(finalImage, hdr)
            hdu.writeto(fileName, overwrite = True)
            print(fileName)
        
        return finalImage
    
    def classic(self, outDir = None, speckle = True, altCCShift = True, ratio = False, selFrac = 0.1):
        """
        
        Basic lucky imaging, first select then shift
        
        speckle - Boolean determines which centering method to use (True uses getSpeckleShift, 
        False uses getShift)
        
        selFrac - Selection fraction, value between 0 and 1
        
        """

        ## Update the header.
        hdr = self.header
        hdr["ALGO"] = ('Classic', 'Algorithm used in the creation of this image')
        hdr["SELFRAC"] = (selFrac, 'Portion (between 0 and 1) of data selected')
        
        #The output filename
        path = PurePosixPath(self.file)
        if outDir is None:
            outDir = self.outDir

        fileName = outDir + path.stem + '_CLASSIC.fits'

        #How do you select - brightest pixel or ratio between the brightest pixel and total? 
        if ratio:
            selIms = self.ratioSelection(self.images, selFrac = selFrac)
        else:
            selIms = self.brightSelection(self.images, selFrac = selFrac)
        numSelIms = len(selIms)
        
        #Select your images and shift them
        selShiftIms = np.zeros([numSelIms, self.imageDim0, self.imageDim1])
        if speckle:
            for i in range(numSelIms):
                selShiftIms[i] = self.getSpeckleShift(selIms[i])
        else:
            for i in range(numSelIms):
                if not altCCShift:
                    selShiftIms[i] = self.getShift(selIms[i]).real
                else:
                    selfShiftIms[i] = self.getShift(selIms[i], selIms[(i-1)*(i!=0)]).real
        shiftAndAdd = np.sum(selShiftIms, axis = 0)
        
        #Normalize to get back to the same units as a single image
        finalImage = shiftAndAdd/numSelIms
        if self.writeTo:
            #Write your new file
            hdu = fits.ImageHDU(finalImage, hdr)
            hdu.writeto(fileName, overwrite = True)
            print("Writing to "+fileName)
        
        return finalImage
    
    # def classicRev(self, outDir, speckle = True, ratio = False, selFrac = 0.1):   
    #     """
    #     DEPRECATED. DO NOT USE. 

    #     First shifts images then selects
    #     Used to investigate phase_cross_correlation centering
    #     phase_cross_correlation found to not work well when images are not selected
    #     first
        
    #     speckle - Boolean determines which centering method to use (True uses getSpeckleShift, 
    #     False uses getShift)
    #     selFrac - Selection fraction, value between 0 and 1
        
    #     """
        
    #     hdr = self.header
        
    #     hdr["ALGO"] = ('Classic Reversed', 'Algorithm used in the creation of this image')
    #     hdr["SELFRAC"] = (selFrac, 'Portion (between 0 and 1) of data selected')
        
    #     path = PurePosixPath(self.file)
    #     fileName = path.stem + '_CLASREV'
        
    #     shiftIms = np.zeros([self.numImages, self.imageDim0, self.imageDim1])
    #     if speckle:
    #         #print('Sp. Shift')
    #         for i in range(self.numImages):
    #             shiftIms[i] = self.getSpeckleShift(self.images[i])
    #     else:
    #         #print('CC Shift')
    #         for i in range(self.numImages):
    #             shiftIms[i] = self.getShift(self.images[i])
    #     if ratio:
    #         selIms = self.ratioSelection(shiftIms, selFrac = selFrac)
    #     else:
    #         selIms = self.brightSelection(shiftIms, selFrac = selFrac)
    #     numSelIms = len(selIms)
    #     selAndAdd = np.sum(selIms, axis = 0)
        
    #     finalImage = selAndAdd/numSelIms
        
    #     hdu = fits.ImageHDU(finalImage, hdr)
        
    #     hdu.writeto(outDir+fileName+'.fits')
        
    #     print(outDir+fileName+'.fits')
        
    #     return finalImage
    
    def shiftAndAdd(self, outDir=None, fileName=None, speckle = True, altCCShift = True):
        
        hdr = self.header
        hdr["ALGO"] = ('Shift and Add', 'Algorithm used in the creation of this image')
        hdr["SELFRAC"] = (1.0, 'Portion (between 0 and 1) of data selected')
        
        path = PurePosixPath(self.file)

        if outDir is None:
            outDir = self.outDir

        if fileName is None:
            fileName = outDir+path.stem + '_SHIFT_ADD.fits'

        shiftIms = np.zeros([self.numImages, self.imageDim0, self.imageDim1])
        if speckle:
            #print('Sp. Shift')
            for i in range(self.numImages):
                shiftIms[i] = self.getSpeckleShift(self.images[i])
        else:
            #print('CC Shift')
            for i in range(self.numImages):
                if not altCCShift:
                    selShiftIms[i] = self.getShift(selIms[i]).real
                else:
                    selfShiftIms[i] = self.getShift(selIms[i], selIms[(i-1)*(i!=0)]).real
        
        finalImage = np.mean(shiftIms, axis = 0)
        if self.writeTo:
            hdu = fits.ImageHDU(finalImage, hdr)
        
            hdu.writeto(fileName, overwrite = True)
            print("Wrote file to {}".format(fileName))
        
        return finalImage
    
    def stack(self, outDir = None, fileName = None):
        hdr = self.header
        hdr["ALGO"] = ('Stack','Algorithm used in the creation of this image')
        hdr["SELFRAC"] = ('1.0','Portion (between 0 and 1) of data selected')
        
        path = PurePosixPath(self.file)

        if outDir is None:
            outDir = self.outDir

        if fileName is None:
            fileName = outDir + path.stem + '_STACK.fits'

        finalImage = np.mean(self.images, axis = 0)
        if self.writeTo:
            hdu = fits.ImageHDU(finalImage, hdr)

            hdu.writeto(fileName, overwrite = True)
        return finalImage

    def classicAssist(self,speckle = True, altCCShift = True, selFrac = 0.1):
        selIms = self.brightSelection(self.images, selFrac = selFrac)
        numSelIms = len(selIms)
        image1 = selIms[0]
        
        selShiftIms = np.zeros([numSelIms, self.imageDim0, self.imageDim1])
        if speckle:
            for i in range(numSelIms):
                selShiftIms[i] = self.getSpeckleShift(selIms[i])
        else:
            for i in range(numSelIms):
                if not altCCShift:
                    selShiftIms[i] = self.getShift(selIms[i]).real
                else:
                    selfShiftIms[i] = self.getShift(selIms[i], selIms[(i-1)*(i!=0)]).real
        shiftAndAdd = np.sum(selShiftIms, axis = 0)
        return shiftAndAdd/numSelIms
    
    def isfasAssist(self, speckle = True, cutoffDenom = 300.0, altCCShift = True, selFrac = 0):
        imageCubeFFT = np.zeros((self.numImages,self.imageDim0,self.imageDim1), dtype = 'complex_')
        imageCubeMag = np.zeros((self.numImages,self.imageDim0,self.imageDim1))
        for i in range(self.numImages):
            tempImage = self.images[i]
            if not speckle:
                if not altCCShift:
                    selShiftIms[i] = self.getShift(selIms[i]).real
                else:
                    selfShiftIms[i] = self.getShift(selIms[i], selIms[(i-1)*(i!=0)]).real
            else:
                tempImageShift = self.getSpeckleShift(tempImage)
            tempImageFFT = np.fft.fftn(tempImageShift)
            imageCubeFFT[i] = np.fft.fftshift(tempImageFFT) 
            tempImageBlur = self.blur(tempImageShift)
            imageCubeMag[i] = tempImageBlur
        
        avgMod = np.zeros([self.imageDim0,self.imageDim1], dtype = 'complex_')
        for i in range(self.numImages):
            avgMod += imageCubeMag[i]
        avgMod /= self.numImages
        avgModMag = np.abs(avgMod)
           
        avgMax = np.amax(avgModMag)
        cutoff = avgMax/cutoffDenom

        indCut = avgModMag < cutoff
        revCut = avgModMag > cutoff
        for i in range(self.numImages):
            imageCubeFFT[i][indCut] = 0

        temparryMag = np.zeros(self.numImages)
        temparryFFT = np.zeros(self.numImages, dtype = 'complex_')
        finalImageFFT = np.zeros((self.imageDim0,self.imageDim1), dtype = 'complex_')
        for i in range(self.imageDim0):
            for j in range(self.imageDim1):
                temparryMag[:] = imageCubeMag[:,i,j]
                temparryFFT[:] = imageCubeFFT[:,i,j]
                ind = self.getHighestValsInd(temparryMag,selFrac)
                finalImageFFT[i][j] = np.mean(temparryFFT[ind])        
        
        finalImage = np.fft.ifftshift(np.abs(np.fft.ifftshift(np.fft.ifftn(finalImageFFT))))
        
        finalImageReturnFFT = np.fft.fftshift(finalImageFFT)
        
        return finalImage, finalImageReturnFFT, revCut
    
    def hybridLI(self, outDir = None, fileName = None, speckle = True, altCCShift = True, selFrac = 0.1):
        
        hdr = self.header
        hdr["ALGO"] = ('Hybrid', 'Algorithm used in the creation of this image')
        hdr["SELFRAC"] = (selFrac, 'Portion (between 0 and 1) of data selected')
        
        path = PurePosixPath(self.file)
        fileName = self.outDir + path.stem + '_HYBRID.fits'
        
        
        classicIm = self.classicAssist(speckle, altCCShift, selFrac)
        isfasRes = self.isfasAssist(speckle, altCCShift, selFrac)
        isfasIm = isfasRes[1]
        isfasIndCut = isfasRes[2]
        
        isfasImShift = np.fft.fftshift(isfasIm)
        classicImFFT = np.fft.fft2(classicIm)
        classicImFFTShift = np.fft.fftshift(classicImFFT)
        classicImFFTShift[isfasIndCut] = 0
        
        
        finalImageFFT = classicImFFTShift + isfasImShift
        
        finalImage = np.fft.ifftshift(np.abs(np.fft.ifftshift(np.fft.ifftn(finalImageFFT))))
        
        if writeTo:
            hdu = fits.ImageHDU(finalImage, hdr)
            hdu.writeto(fileName)
            print("Writing to "+fileName)
        
        return finalImage


