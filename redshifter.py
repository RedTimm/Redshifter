#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shifts an image to a different (higher) redshift
By Roland Timmerman
"""


#Imports
import os
import errno
import glob
import copy
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import convolve, Gaussian2DKernel
from FITS_tools.hcongrid import hcongrid


def inputchecker(args):
    """
    Checks validity of user input
    """
    
    if args['modelimage'] == None:
        raise Exception("No input model given")
    
    if not glob.glob(args['modelimage']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args['modelimage'])
    
    if args['z_in'] == None:
        raise Exception("No input redshift given")
    
    if args['z_out'] == None:
        raise Exception("No output redshift given")
    
    if args['z_out'] < args['z_in']:
        raise Exception("Unable to move to a lower redshift")
    
def redshift_image(args):
    """
    Moves the input image virtually to a higher redshift
    """
    
    #Get header info
    hdu_list = fits.open(args['modelimage'])
    input_header = hdu_list[0].header
    input_model = hdu_list[0].data[0,0,:,:]

    #Remove excess axes (necessary evil)
    wcs = WCS(input_header)
    wcs = wcs.dropaxis(2).dropaxis(2)
    input_header.update(wcs.to_header())
    input_header.set('NAXIS', 2)
    input_header.remove('NAXIS3', ignore_missing=True)
    input_header.remove('CTYPE3', ignore_missing=True)
    input_header.remove('CRPIX3', ignore_missing=True)
    input_header.remove('CRVAL3', ignore_missing=True)
    input_header.remove('CDELT3', ignore_missing=True)
    input_header.remove('CUNIT3', ignore_missing=True)
    input_header.remove('NAXIS4', ignore_missing=True)
    input_header.remove('CTYPE4', ignore_missing=True)
    input_header.remove('CRPIX4', ignore_missing=True)
    input_header.remove('CRVAL4', ignore_missing=True)
    input_header.remove('CDELT4', ignore_missing=True)
    input_header.remove('CUNIT4', ignore_missing=True)
    
    #Define cosmology
    cosmo = FlatLambdaCDM(H0=args['H0'], Om0=args['Omega_m0'])
    
    #Calculate angular scale difference between the two redshifts
    z_in_Da = cosmo.angular_diameter_distance(args['z_in'])
    z_out_Da = cosmo.angular_diameter_distance(args['z_out'])
    z_in_Dl = cosmo.luminosity_distance(args['z_in'])
    z_out_Dl = cosmo.luminosity_distance(args['z_out'])
    
    scale_difference = float(z_in_Da/z_out_Da)
    flux_difference = float(z_in_Dl/z_out_Dl)**2 * ((1+args['z_in'])/(1+args['z_out']))**(-args['alpha']-1)
    
    virtual_header = copy.deepcopy(input_header)
    
    #Figure out which are the RA and Dec axes
    RA_idx = 0
    Dec_idx = 0
    for idx in range(1, 1+virtual_header['NAXIS']):
        if virtual_header[f'CTYPE{idx}'][:2] == "RA":
            RA_idx = idx
        if virtual_header[f'CTYPE{idx}'][:3] == "DEC":
            Dec_idx = idx
    if RA_idx == 0 or Dec_idx == 0:
        raise Exception("Unable to identify RA and Dec axes")
    
    RA_pixel_scale = virtual_header[f'CDELT{RA_idx}']
    Dec_pixel_scale = virtual_header[f'CDELT{Dec_idx}']
    
    pixel_size = copy.deepcopy(RA_pixel_scale)
    
    #Regrid image to virtual higher redshift
    RA_pixel_scale /= scale_difference
    Dec_pixel_scale /= scale_difference
        
    virtual_header[f'CDELT{RA_idx}'] = RA_pixel_scale
    virtual_header[f'CDELT{Dec_idx}'] = Dec_pixel_scale
    
    virtual_model = hcongrid(input_model, input_header, virtual_header)
        
    #Set correct flux density
    virtual_model *= np.sum(input_model)*flux_difference/np.sum(virtual_model)
    
    #Obtain convolution kernel
    if args['dirtybeam'] is not None:
        hdu_list_dirtybeam = fits.open(args['dirtybeam'])
        beam = hdu_list_dirtybeam[0].data[0,0,:,:]
    else:
        if args['bmaj'] is not None:
            bmaj = args['bmaj']/3600/pixel_size/2.35482
        else:
            bmaj = input_header['BMAJ']/pixel_size/2.35482
        if args['bmin'] is not None:
            bmin = args['bmin']/3600/pixel_size/2.35482
        else:
            bmin = input_header['BMIN']/pixel_size/2.35482
        if args['bpa'] is not None:
            bpa = args['bpa']/180*3.14159265358979
        else:
            bpa = input_header['BPA']/180*3.14159265358979
        
        #Create Gaussian kernel
        beam = Gaussian2DKernel(bmin, bmaj, bpa, x_size=31, y_size=31)
        
    #Convolve model with beam
    virtual_image = convolve(virtual_model, beam, normalize_kernel=False)*2*3.14159265358979*bmin*bmaj
    
    #Obtain background noise
    if args['rmsmap'] is not None:
        rms_hdu = fits.open(args['rmsmap'])
        rms_map = rms_hdu[0].data[0,0,:,:]
    else:
        img_dim = (input_header[f'NAXIS{RA_idx}'], input_header[f'NAXIS{Dec_idx}'])
        rms_base = np.random.normal(0, 1, size=img_dim)
        rms_map = convolve(rms_base, beam, normalize_kernel=False)
        rms_map *= args['rmslevel']/np.std(rms_map)
        
    #Add rms to source image
    virtual_image += rms_map
        
    #Write virtual image to file
    try:
        if args['outname'][-5:] != ".fits":
            args['outname'] += ".fits"
    except IndexError:
        args['outname'] += ".fits"
    hdu = fits.PrimaryHDU(data=virtual_image, header=input_header)
    hdu.writeto(args['outname'], overwrite=True)


if __name__=="__main__":
    
    #Set up input parser
    parser = argparse.ArgumentParser(description=\
        '''
        Shifts an image in redshift
        ''', formatter_class=argparse.RawTextHelpFormatter)
    
    #Required parameters
    parser.add_argument('--modelimage', help="(Required) Clean components to map towards higher redshift", type=str)
    parser.add_argument('--z_in', help="(Required) Redshift of the input image", type=float)
    parser.add_argument('--z_out', help="(Required) Target redshift to move the image to", type=float)

    #Cosmological parameters
    parser.add_argument('--H0', help="Present-day Hubble parameter (in km/s/Mpc)", default=70.0, type=float)
    parser.add_argument('--Omega_m0', help="Present-day cosmological mass density", default=0.3, type=float)

    #Additional parameters
    parser.add_argument('--alpha', help="Spectral index to assume for the entire source, to account for redshifting", default=-0.7, type=float)
    parser.add_argument('--bmaj', help="Major axis of the clean beam to restore the model with (arcseconds). Default is to read from input fits", type=float)
    parser.add_argument('--bmin', help="Minor axis of the clean beam to restore the model with (arcseconds). Default is to read from input fits", type=float)
    parser.add_argument('--bpa', help="Position angle of the clean beam to restore the model with (degrees). Default is to read from input fits", type=float)
    parser.add_argument('--dirtybeam', help="If given, will convolve the model image with this dirty beam instead of a clean beam", type=str)
    parser.add_argument('--outname', help="Output name for the virtual image", default="virtual_image.fits", type=str)
    parser.add_argument('--rmslevel', help="If no rms map given to superimpose the image on (see --rmsmap), create artificial rms map with this noise level", default=0.00005, type=float)
    parser.add_argument('--rmsmap', help="rms map to superimpose the image on. If not given, will attempt to simulate rms map (see --rmslevel)", type=str)

    #Parse inputs
    options = parser.parse_args()
    args = vars(options)
        
    #Verify that user inputs are valid
    inputchecker(args)
    
    #Do the redshifting
    redshift_image(args)