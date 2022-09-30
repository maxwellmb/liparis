from astropy.io import fits
import numpy as np



class Datacube():
    '''
    A datacube object
    '''
    def __init__(self,filename, ext_no = 1):
        
        #Read in the data
        hdul = fits.open(filename)
        # self.data = hdul[ext_no].data
        self.data = np.array([hdu.data for hdu in hdul[1:]]) #Just for Kyle's weird data
        self.header = hdul[ext_no].header

        self.nframes = self.data.shape[0]

    def align_images(self, images, method = "peak"):
        '''
        Align the image stack

        Args: 
        Method - You can align with the peak pixel or via a fourier alignemnet 
                    technique
        '''


        self.aligned_data

    def select_lucky_images(self, method = "peak",selection_fraction = 0.1,
                        nframes_to_select = None):
        '''
        A function that gets the indices of 

        methods could be peak pixel value, or some kind of fourier selection

        Args: 
        Method - The method to select the frames [str]
        selectrion_fraction - what fraction of the frames do you want to select [float]
        nframes_to_select - how many frames do you want to select? Supercedes selectrion_fraction [int]
        '''
        
        #If nframes_to_selct is passed, it supercedes selection_fraction 
        if nframes_to_select is None:
            nframes_to_select = int(np.floor(self.nframes*selection_fraction))

        #What method will we use to the select the good frames. 
        if method == "peak":
            peak_values = [np.max(data_frame) for data_frame in self.data]

            sorted_indices = np.argsort(peak_values)

            selected_indices = sorted_indices[-nframes_to_select:] #Only pick the highest ones

            self.lucky_indices = selected_indices
        elif method == "fourier":
            print("Fourier image selection is not yet implemented")
        else: 
            print("Your method of {} is not yet implemented")


    def construct_lucky_image(self,method="median"):
        '''
        Build a lucky image based on the selected indices
        '''

        if method == "median":
             self.lucky_image = np.median(self.data[self.lucky_indices],axis=0)

        
       
