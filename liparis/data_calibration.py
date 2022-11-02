## -- IMPORTS
from astropy.io import fits
from astropy.stats import sigma_clip
import glob
import matplotlib.pyplot as plt
import numpy as np

## -- FUNCTIONS

def build_dark(dark_path, norm_to_exposure=False, writeout=None):
    
    super_dark = median_combine(dark_path)
    super_dark_subtracted = super_dark - bias_image
    
    dark_exp_time = fits.getheader(glob.glob(dark_path)[0])['SAMTIME'] if norm_to_exposure else 1 # what was this?
    super_dark /= dark_exp_time 
    
    if writeout is not None:
        hdu_list = fits.HDUList([fits.PrimaryHDU(super_dark)])
        hdu_list.writeto(writeout, overwrite=True)

    return super_dark


def build_flat(flat_path, dark_image):
    
    flat = median_combine(flat_path, norm_to_exposure=True, norm_to_1=True, subtract=dark_image)

    if writeout is not None:
        hdu_list = fits.HDUList([fits.PrimaryHDU(flat)])
        hdu_list.writeto(writeout, overwrite=True)

    return flat


def median_combine(path, norm_to_exposure=False, norm_to_1=False, subtract=None):
    """ Median combines a given set of images. Some boolean flags because flats
    are complicated.
    
    Parameters
    ----------
    path : str
        Glob-able path of files.
    norm_to_exposure : bool
        Whether or not to normalize by exposure time before median combining. 
    norm_to_1 : bool
        Whether or not to normalize to 1 before median combining.
    subtract : np.array or None
        Dark to subtract off before combining, or None.

    Returns
    -------
    median : np.array
        Array of median combined images.
    """

    files = glob.glob(path)
    size = np.shape(fits.getdata(files[0]))
    
    cube = np.zeros((size[0], size[1], len(files)))
    for index, fits_file in enumerate(files):

        with fits.open(fits_file) as hdu:
            data = hdu[0].data
            exp_time = hdu[0].header['SAMTIME'] # AGAIN

        # Extra fiddling according to flags
        data = data/exp_time if norm_to_exposure else data
        data = data - subtract if subtract is not None else data
        sigma_max = np.max(sigma_clip(data, masked=False, axis=None))
        data = data/sigma_max if norm_to_1 else data
        
        cube[:, :, index] = data

    median = np.median(cube, axis=2)

    return median

def reduce_data(science_path, dark_path, flat_path, data_path_out):
        """ Builds the final reduced science image. 

    Parameters
    ----------
    science_path: str
        Glob-able path to the science fits files.
    dark_path : np.array
        Array of super dark data.
    flat_path : np.array
        Dark subtracted and time averaged flat data.
    writeout : str or None
        Name of the fits file to write out, if at all.
    data_path_out : str or None
        Name of the data to write out, if at all.

    Returns
    -------
    reduced_science : np.array
        Array of final reduced science image.
    """

    super_science = median_combine(science_image_path)
    dark_image = build_dark(dark_path)
    flat_image = build_flat(flat_path, dark_image)

    #science_exp_time = fits.getheader(glob.glob(science_image_path)[0])['EXPTIME']

    reduced_science = (super_science - dark_image)/flat_image

    if data_path_out is not None:
        hdu_list = fits.PrimaryHDU([reduced_science])
        hdu_list.writeto(data_path_out)

    return reduced_science


## -- RUN
if __name__=="__main__":
    
    science_path = 'path_to_orkid_raw_files_*.fits'
    dark_path = 'path_to_darks_of_same_exposure_*.fits'
    flat_path = 'path_to_flats_*.fits'

    data_path_out = 'test_reduced_data.fits'

    reduce_data(science_path, dark_path, flat_path, data_path_out)

    
