r"""
author:     Bernhard Hörl
version:    0.1
date:       2019-03-15

Inspired by the python scripts from Stefan Meingast.

MasterMetry
===========

TODO
"""

import warnings
import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import make_lupton_rgb
from photutils import (DAOStarFinder, CircularAperture, CircularAnnulus,
                       aperture_photometry)

warnings.filterwarnings("ignore")


def master_bias(file_list, dtype=np.float32, saveto=None, overwrite=False):
    r"""
    Generate the master bias from a list of bias files.

    Parameters
    ----------
    file_list : list(str)
        List of file names.
    dtype : data-type, optional
        Type to use for computing, defaults to np.float32.
    saveto : str, optional
        If this is not set (None, default) files won't be saved.
        If this is set to a string, save the file with the string as name.
    overwrite : bool, optional
        While saving, overwrite existing files? Defaults to False.

    Returns
    -------
    hdul : astropy.io.fits.HDUList
    """
    n = len(file_list)
    data = None
    for file in file_list:
        with fits.open(file) as hdul:
            if data is None:
                header = hdul[0].header
                data = hdul[0].data.astype(dtype)
            else:
                data += hdul[0].data.astype(dtype)
    data /= n
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul[0].header = header
    if saveto is not None:
        hdul.writeto(saveto, overwrite=overwrite)
    return hdul


def master_dark(file_list, master_bias_hdul, dtype=np.float32, saveto=None,
                overwrite=False):
    r"""
    Generate the master dark from a list of dark files,
    a master bias file is necessary for computation.

    Parameters
    ----------
    file_list : list(str)
        List of file names.
    master_bias_hdul : astropy.io.fits.HDUList
        Calculated master bias.
    dtype : data-type, optional
        Type to use for computing, defaults to np.float32.
    saveto : str, optional
        If this is not set (None, default) files won't be saved.
        If this is set to a string, save the file with the string as name.
    overwrite : bool, optional
        While saving, overwrite existing files? Defaults to False.

    Returns
    -------
    hdul : astropy.io.fits.HDUList
    """
    n = len(file_list)
    master_bias = master_bias_hdul[0].data.astype(dtype)
    data = np.zeros((n, *master_bias.shape), dtype)
    total_exposure_time = None
    for i, file in enumerate(file_list):
        with fits.open(file) as hdul:
            if total_exposure_time is None:
                header = hdul[0].header
                total_exposure_time = exposure_time = header["EXPOSURE"]
            else:
                total_exposure_time += hdul[0].header["EXPOSURE"]
            data[i] = hdul[0].data.astype(dtype) - master_bias
    data = np.median(data, axis=0) / exposure_time
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul[0].header = header
    hdul[0].header["EXPTIME"] = total_exposure_time
    if saveto is not None:
        hdul.writeto(saveto, overwrite=overwrite)
    return hdul


def master_flat(file_list, master_bias_hdul, master_dark_hdul,
                dtype=np.float32, bit=16, saveto=None, overwrite=False):
    r"""
    Generate the master flat from a list of flat files,
    master bias and master dark files are necessary for computation.
    Has to be done for each filter individually.

    Parameters
    ----------
    file_list : list(str)
        List of file names.
    master_bias_hdul : astropy.io.fits.HDUList
        Calculated master bias.
    master_dark_hdul : astropy.io.fits.HDUList
        Calculated master dark.
    dtype : data-type, optional
        Type to use for computing, defaults to np.float32.
    bit : int, optional
        Used to determine the maximum ADU value (2^bit - 1) of the initial
        fits files, which gets overwritten when generating new files with a
        different data type.
    saveto : str, optional
        If this is not set (None, default) files won't be saved.
        If this is set to a string, save the file with the string as name.
    overwrite : bool, optional
        While saving, overwrite existing files? Defaults to False.

    Returns
    -------
    hdul : astropy.io.fits.HDUList
    """
    n = len(file_list)
    master_bias = master_bias_hdul[0].data.astype(dtype)
    master_dark = master_dark_hdul[0].data.astype(dtype)
    data = None
    total_exposure_time = None
    for file in file_list:
        with fits.open(file) as hdul:
            exposure_time = hdul[0].header["EXPOSURE"]
            flat = hdul[0].data.astype(dtype)
            if total_exposure_time is None:
                header = hdul[0].header
                total_exposure_time = exposure_time
                data = flat - master_bias - exposure_time * master_dark
            else:
                total_exposure_time += exposure_time
                data = flat - master_bias - exposure_time * master_dark
    data /= n
    min_val = np.min(data)
    if min_val < 0:
        data -= min_val
    data /= 2 ** bit - 1  # 65535 ADU for 16 bit
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul[0].header = header
    hdul[0].header["EXPTIME"] = total_exposure_time
    if saveto is not None:
        hdul.writeto(saveto, overwrite=overwrite)
    return hdul


def normalize(data, method):
    r"""
    Normalizes the image ('removes background noise').

    Parameters
    ----------
    data : np.ndarray
        Data which should be normalized.
    method : str
        Method to use for normalization.
        Valid values/metods are 'mean' or 'median'.

    Returns
    -------
    data : np.ndarray
    """
    if method is None:
        return data
    elif method == "mean":
        return np.divide(data, np.nanmean(data))
    elif method == "median":
        return np.divide(data, np.nanmedian(data))
    else:
        raise ValueError("Only 'mean' or 'median' are possible methods.")


def master_science(file_list, master_bias_hdul, master_dark_hdul,
                   master_flat_hdul, dtype=np.float32, gain=None, method=None,
                   saveto=None, overwrite=False):
    r"""
    Generate the science frame out of master bias, master dark, master flat
    files and a list of light-frame file names.
    Has to be done for each filter individually.

    Parameters
    ----------
    file_list : list(str)
        List of file names.
    master_bias_hdul : astropy.io.fits.HDUList
        Calculated master bias.
    master_dark_hdul : astropy.io.fits.HDUList
        Calculated master dark.
    master_flat_hdul : astropy.io.fits.HDUList
        Calculated master flat for a filter.
    dtype : data-type, optional
        Type to use for computing, defaults to np.float32.
    gain : float, optional
        If not none, use this value for the gain instead of the one present
        in the fits table.
    method : str, optional
        Method to use for normalization if value is not None.
        Valid values/metods are 'mean' or 'median'.
    saveto : str, optional
        If this is not set (None, default) files won't be saved.
        If this is set to a string, save the file with the string as name.
    overwrite : bool, optional
        While saving, overwrite existing files? Defaults to False.

    Returns
    -------
    hdul : astropy.io.fits.HDUList
    """
    master_bias = master_bias_hdul[0].data.astype(dtype)
    master_dark = master_dark_hdul[0].data.astype(dtype)
    master_flat = master_flat_hdul[0].data.astype(dtype)
    data = None
    total_exposure_time = None
    for file in file_list:
        with fits.open(file) as hdul:
            exposure_time = hdul[0].header["EXPOSURE"]
            if total_exposure_time is None:
                header = hdul[0].header
                if gain is None:
                    gain = header["EGAIN"]
                total_exposure_time = exposure_time
                data = (hdul[0].data.astype(dtype) - master_bias -
                        exposure_time * master_dark) / master_flat
            else:
                total_exposure_time += exposure_time
                data += (hdul[0].data.astype(dtype) - master_bias -
                         exposure_time * master_dark) / master_flat
    data *= gain / total_exposure_time
    data = normalize(data, method)
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul[0].header = header
    hdul[0].header["EXPTIME"] = total_exposure_time
    if saveto is not None:
        hdul.writeto(saveto, overwrite=overwrite)
    return hdul


def master_color(R_hdul, G_hdul, B_hdul, dtype=np.float32, saveto=None,
                 **kwargs):
    r"""
    Generate an RGB image based on three science frames (preferable R, V, B).
    This use `astropy.visualization.make_lupton_rgb`.

    Parameters
    ----------
    R_hdul : astropy.io.fits.HDUList
        Used for red channel.
    G_hdul : astropy.io.fits.HDUList
        Used for green channel (normally use V-filter).
    B_hdul : astropy.io.fits.HDUList
        Used for blue channel.
    dtype : data-type, optional
        Type to use for computing, defaults to np.float32.
    saveto : str, optional
        If this is not set (None, default) files won't be saved.
        If this is set to a string, save the file with the string as name.
        The format is determined by the file extension.
        Files are ALWAYS OVERWRITTEN!

    For other parameters see astropy.visualization.make_lupton_rgb.

    Returns
    -------
    data : numpy.ndarray
    """
    data = np.array(
        [R_hdul[0].data.astype(dtype),
         G_hdul[0].data.astype(dtype),
         B_hdul[0].data.astype(dtype)],
        dtype
    )
    return make_lupton_rgb(*data, filename=saveto, **kwargs)


def image_offset(image, x=0, y=0, dtype=np.float32):
    r"""
    Move an image (numpy.ndarray) on the x and y plane,
    out of bound data will be set to zero.
    Floats will be rounded and converted to int.

    Parameters
    ----------
    image : numpy.ndarray
        Image to transform.
    x : int, optional
        Move image on the x-axis from left to right if x is positive
        and vice-versa.
    y : int, optional
        Move image on the y-axis from top to bottom if y is positive
        and vice-versa.
    dtype : data-type, optional
        Type to use for computing, defaults to np.float32.

    Returns
    -------
    data : numpy.ndarray

    Note
    ----
    This has only been tested for square images, and the way I have
    implemented this is far from elegant.
    I just changed signs and positions until everything moved in the
    desired direction.
    """
    x = int(round(x, 0))
    y = int(round(y, 0))
    resolution = image.shape
    new = np.zeros(resolution, dtype=dtype)
    A = [[0, 0],
         [0, 0]]
    if x > 0:
        A[0] = [0, -x]
    elif x < 0:
        A[0] = [x, 0]
    if y > 0:
        A[1] = [0, -y]
    elif y < 0:
        A[1] = [y, 0]
    # I have no idea what I am doing.
    for j in np.arange(A[0][1], resolution[0]+A[0][0]):
        for i in np.arange(A[1][1], resolution[1]+A[1][0]):
            new[i+A[0][0], j+A[1][0]] = image[i+A[0][1], j+A[1][1]]
    return new


def master_photometry(hdul, dtype=np.float32, method=None, sigma=3, fwhm=4,
                      kappa=5, r=3, r_in=5, r_out=9, save_format="fits",
                      saveto=None, overwrite=False):
    r"""
    Generate statistics, positions and more fore a given hdul object.
    This uses photutils package, see it for more information,
    also astropy tables could be relevant to look at.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Image to process.
    dtype : data-type, optional
        Type to use for computing, defaults to np.float32.
    method : str, optional
        Method to use for normalization if value is not None.
        Valid values/metods are 'mean' or 'median'.
        Pay attention if you already used this with master_science,
        it's the same thing!
    sigma : float, optional
        Sigma level for mean, median and standard deviation, defaults to 3.
    fwhm : float, optional
        Expected FWHM in pixels, defaults to 4.
    kappa : float, optional
        Sigma level for source detection above background, defaults to 5.
    r : float, optional
        Aperture radius, defaults to 3.
    r_in : float, optional
        Inner sky annulus radius, defaults to 5.
    r_out : float, optional
        Outer sky annulus radius, defaults to 9.
    save_format : str, optional
        Format to save the table with, defaults to 'fits',
        has no effect if saveto is None.
    saveto : str, optional
        If this is not set (None, default) files won't be saved.
        If this is set to a string, save the file with the string as name.
    overwrite : bool, optional
        While saving, overwrite existing files? Defaults to False.

    Returns
    -------
    table : photutils.aperture_photometry
    """
    data = normalize(hdul[0].data.astype(dtype), method)
    mean, median, std = sigma_clipped_stats(data, sigma=sigma)
    DAOfind = DAOStarFinder(fwhm=fwhm, threshold=kappa*std)
    positions = DAOfind(data-median)
    centroid = (positions["xcentroid"], positions["ycentroid"])
    aperture = CircularAperture(centroid, r)
    annulus = CircularAnnulus(centroid, r_in, r_out)
    apertures = [aperture, annulus]
    table = aperture_photometry(data, apertures)
    bg_mean = table["aperture_sum_1"] / annulus.area()
    bg_sum = bg_mean * aperture.area()
    table["residual_aperture_sum"] = table["aperture_sum_0"] - bg_sum
    table["mag"] = - 2.5 * np.log10(table["residual_aperture_sum"])
    if saveto is not None:
        # TODO: dtype?
        table.write(saveto, overwrite=overwrite, format=save_format)
    return table


class Master:
    r"""
    This is a auto wrapper for all previous functions.
    Just provide file list for all the fits that should (and must) be used,
    the type is automatically selected based on the fits header.
    All mandatory files (bias, dark, flat, light) need to be in the file list,
    or the calculations wont be successful.

    Parameters
    ----------
    file_list : list(str)
        List of file names, glob may come in handy.
    dtype : data-type, optional
        Type to use for computing, defaults to np.float32.
    bit : int, optional
        Used to determine the maximum ADU value (2^bit - 1) of the initial
        fits files, which gets overwritten when generating new files with a
        different data type (see master_flat).
    gain : float, optional
        If not none, use this value for the gain instead of the one present
        in the fits table (see master_science).
    method : str, optional
        Method to use for normalization if value is not None.
        Valid values/metods are 'mean' or 'median'.
    save_dir : str, optional
        Specify the directory in which the generated fits files should be
        saved, defaults to '.' (current directory).
        Has no effect if save_all is False.
    save_all : bool, optional
        Save all generated hdul objects to fits files when set to True,
        default is False.
    overwrite : bool, optional
        While saving, overwrite existing files? Defaults to False.
    """
    def __save(self, data_type, filter_type=""):
        if not self.__save_all:
            return None
        elif data_type == "B":
            file_name = "master_bias"
        elif data_type == "D":
            file_name = "master_dark"
        elif data_type == "F":
            file_name = f"master_flat_{filter_type}"
        elif data_type == "S":
            file_name = f"science_frame_{filter_type}"
        return self.__save_dir + file_name + self.__extension

    def __init__(self, file_list, dtype=np.float32, bit=16, gain=None,
                 method=None, save_dir=".", save_all=False, overwrite=False):
        self.__dtype = dtype
        self.__save_all = save_all
        if save_dir[-1] != r"/":
            save_dir += r"/"
        self.__save_dir = save_dir
        self.__extension = ".fits"
        self.offset_history = []
        file_list = sorted(file_list)
        bias_files = []
        dark_files = []
        flat_files = {}
        light_files = {}
        for file_name in file_list:
            with fits.open(file_name) as hdul:
                image_type = hdul[0].header["IMAGETYP"].lower()
                if "bias" in image_type:
                    bias_files.append(file_name)
                elif "dark" in image_type:
                    dark_files.append(file_name)
                elif "flat" in image_type:
                    filter_type = hdul[0].header["FILTER"]
                    if filter_type in flat_files:
                        flat_files[filter_type].append(file_name)
                    else:
                        flat_files[filter_type] = [file_name]
                elif "light" in image_type:
                    filter_type = hdul[0].header["FILTER"]
                    if filter_type in light_files:
                        light_files[filter_type].append(file_name)
                    else:
                        light_files[filter_type] = [file_name]
                else:
                    raise IndexError(f"Image type '{image_type}' unknown.")
        self.bias = master_bias(
            bias_files,
            dtype,
            self.__save("B"),
            overwrite
        )
        self.dark = master_dark(
            dark_files,
            self.bias,
            dtype,
            self.__save("D"),
            overwrite
        )
        self.flat = {}
        self.science = {}
        for filter_type in flat_files.keys():
            if filter_type not in light_files.keys():
                raise IndexError("Flat & light have different filter(s).")
            self.flat[filter_type] = master_flat(
                flat_files[filter_type],
                self.bias,
                self.dark,
                dtype,
                bit,
                self.__save("F", filter_type),
                overwrite
            )
            self.science[filter_type] = master_science(
                light_files[filter_type],
                self.bias,
                self.dark,
                self.flat[filter_type],
                dtype,
                gain,
                method,
                self.__save("S", filter_type),
                overwrite
            )

    def normalize(self, method):
        r"""
        Wrapper for the normalize function applied on the science frames.
        """
        for filter_type in self.science.keys():
            self.science[filter_type][0].data = normalize(
                self.science[filter_type][0].data.astype(self.__dtype),
                method
            )

    def offset(self, filter_type, x=0, y=0, return_only=False):
        r"""
        Wrapper for image_offset.
        Attention, data gets lost if the image moves out of bounds.

        Parameters
        ----------
        filter_type : str
            Filter based of the science frame to process, only pass a single
            character.
        return_only : bool, optional
            Do not apply the offset on the object itself, just return
            the result, defaults to False.

        For other options see image_offset.
        """
        new_image = image_offset(
            self.science[filter_type][0].data.astype(self.__dtype),
            x=x,
            y=y,
            dtype=self.__dtype
        )
        if return_only:
            return new_image
        else:
            self.offset_history.append((x, y))
            self.science[filter_type][0].data = new_image

    def export_Tables(self, output_file, method=None, sigma=3, fwhm=4,
                      kappa=5, r=3, r_in=5, r_out=9, overwrite=False):
        r"""
        Wrapper for master_photometry.

        Parameters
        ----------
        output_file : str
            Use %s for the filter type, the save format will be determined
            by the file extensions.

        For other options see master_photometry.
        """
        save_format = output_file.split(".")[-1]
        for filter_type in self.science.keys():
            master_photometry(
                self.science[filter_type],
                self.__dtype,
                method,
                sigma,
                fwhm,
                kappa,
                r,
                r_in,
                r_out,
                save_format,
                output_file % filter_type,
                overwrite
            )

    def export_RGB(self, filter_order, saveto=None, **kwargs):
        r"""
        Wrapper for master_color, but instead of giving the R, G, and B hdul
        objects, an (ordered) string with length 3 where each letter
        represents the filter is used, keep the order R-G-B in mind.

        Parameters
        ----------
        filter_order : str
            String with length 3, for example 'RGB' red-green-blue
            (order is important) for the corresponding color channels.
            Most of the time 'RVB' will be used.

        For other options see master_color.
        """
        if len(self.science) < 3:
            raise FileNotFoundError("A RGB image needs 3 science frames.")
        if (not isinstance(filter_order, str)) or (len(filter_order) != 3):
            raise ValueError("Not a string with length 3.")
        for filter_type in filter_order:
            if filter_type not in self.science.keys():
                raise ValueError(f"Filter '{filter_type}' not found.")
        self.RGB = master_color(
            self.science[filter_order[0]],
            self.science[filter_order[1]],
            self.science[filter_order[2]],
            self.__dtype,
            saveto,
            **kwargs
        )
        return self.RGB
