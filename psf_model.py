import numpy as np
import sys
from astropy.io import fits
from astropy.wcs import WCS 
import glob
from scipy.interpolate import griddata

import cv2,csv
import time
import copy
import seaborn
import photutils.psf as pt
from matplotlib.patches import Rectangle
from astropy.table import Table,Column
import matplotlib.pyplot as plt
from photutils.psf import (IterativelySubtractedPSFPhotometry,BasicPSFPhotometry,FittableImageModel)
from photutils.detection import IRAFStarFinder,DAOStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm,sigma_clipped_stats
from photutils import find_peaks
from photutils.segmentation import make_source_mask
from photutils import CircularAnnulus,CircularAperture,aperture_photometry
from astropy.nddata import NDData
from photutils.psf import extract_stars,EPSFStars
from photutils import EPSFBuilder
from astropy.visualization import simple_norm
from astropy.nddata import Cutout2D

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PLOTTING FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_extracted_stars(stars):

	nrows = int(len(stars)/5)
	ncols = 6
	size = 350.0
	#nrows = 1
	#ncols = 4

	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
	                       squeeze=True)
	ax = ax.ravel()
	for i in range(len(stars)):
		norm = simple_norm(stars[i], 'log', percent=99.)
		ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
		ax[i].vlines(size/2.,0,size,colors='r',linestyles='dashed')
		ax[i].hlines(size/2.,0,size,colors='r',linestyles='dashed')
		ax[i].set_xlim(125,225)
		ax[i].set_ylim(125,225)

	plt.savefig('all_extracted_test.pdf')
	#plt.show()
	plt.close()

	return

def plot_extracted_stars_single_targets(star,filename):	

	folder = filename[0].split('/')[0]
	file = filename[0].split('/')[1]

	fig,ax = plt.subplots()
	norm = simple_norm(star[0], 'log', percent=99.)
	ax.imshow(star[0], norm=norm, origin='lower', cmap='viridis')

	plt.savefig('all_extracted.pdf')
	plt.close()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_epsf(epsf):

	norm = simple_norm(epsf.data, 'log', percent=99.)
	plt.figure()
	plt.imshow(epsf.data, norm=norm, origin='lowerleft', cmap='viridis')
	plt.colorbar()
	plt.show()

	return


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SIMPLE GAUSSIAN PSF MODEL AND ITERATIVELY FITTING MODEL
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def simple_psf_options():

	psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
	
	psf_model.x_0.fixed = True
	psf_model.y_0.fixed = True
	pos = Table(names=['x_0', 'y_0'], data=[xcent,ycent])
	
	photometry = BasicPSFPhotometry(group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=psf_model,fitter=LevMarLSQFitter(),fitshape=(11,11))
	
	photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=psf_model,fitter=LevMarLSQFitter(),niters=1, fitshape=(11,11))
	result_tab = photometry(image=imdata)
	residual_image = photometry.get_residual_image()
	
	plt.figure()
	plt.imshow(residual_image,vmin=-0.2,vmax=2.0,cmap='plasma')
	plt.show()

	return


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# INITIAL SETUP FUNCTION
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def init_setup(filename):

	fitimage = fits.open(str(filename))
	imdata = fitimage[1].data
	head = fitimage[0].header

	fitimagevis = fits.open('../hst_data/serpens3/idxq28030_drc.fits')
	imdata_vis = fitimagevis[1].data
	head_vis = fitimagevis[0].header
	
	bkgrms = MADStdBackgroundRMS()
	std = bkgrms(imdata)
	mean = np.mean(imdata)
	sigma_psf = 2.0
	iraffind = IRAFStarFinder(threshold=3.5*std,fwhm=sigma_psf*gaussian_sigma_to_fwhm,minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,sharplo=0.0, sharphi=2.0)
	
	daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
	mmm_bkg = MMMBackground()

	return imdata,imdata_vis, bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# EPSF FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def construct_epsf(filename,starsn,imdata,mean):

	if 'alls' in starsn:
		print('True')
		maxf = 1.0
	else:
		maxf = 0.5

	peaks_tbl = find_peaks(imdata,threshold=maxf)
	peaks_tbl['peak_value'].info.format = '%.8g'
	print(peaks_tbl)

	peaks_tbl.remove_row(0)

	plt.figure()
	plt.scatter(peaks_tbl['x_peak'],peaks_tbl['y_peak'],c='k')
	plt.imshow(imdata,vmin=-0.2,vmax=2.,origin='lowerleft')
	plt.show()

	stars_tbl = Table()
	stars_tbl['x'] = peaks_tbl['x_peak']
	stars_tbl['y'] = peaks_tbl['y_peak']

	mean_val, median_val, std_val = sigma_clipped_stats(imdata,sigma=2.0)
	imdata -= median_val

	nddata = NDData(data=imdata)
	#epsf constructed with be 4x this size
	stars = extract_stars(nddata,stars_tbl,size=20)

	epsf_builder = EPSFBuilder(oversampling=4,maxiters=3,progress_bar=False)
	epsf,fitted_stars = epsf_builder(stars)

	saveepsf = False

	if saveepsf:
		hdu = fits.PrimaryHDU(epsf.data)
		hdul = fits.HDUList([hdu])
		if 'alls' in starsn:
			print('True')
			hdul.writeto(str(filename)+'_epsf_all.fits')
		else: 
			hdul.writeto(str(filename)+'_epsf_lim_0_not_target_cutout.fits')

	return stars, epsf


def highbin_epsf(imdata):
	# change to 149 and 150, bascially add in a 1 
	# REBIN TO A FINER PIXEL SCALE - 10x AS HIGH RESOLUTION
	og_x,og_y = np.mgrid[0:149:150j, 0:149:150j]
	new_x,new_y = np.mgrid[0:149:1500j, 0:149:1500j]

	XYpairs = np.dstack([og_x,og_y]).reshape(-1, 2)

	grid0 = griddata(XYpairs,imdata.flatten(),(new_x,new_y),method='cubic')

	return grid0

def highbin_im(imdata):

	imdata = np.nan_to_num(imdata)

	og_x,og_y = np.mgrid[0:517:518j, 0:580:581j]
	new_x,new_y = np.mgrid[0:517:5180j, 0:580:5810j]

	XYpairs = np.dstack([og_x,og_y]).reshape(-1, 2)

	grid0 = griddata(XYpairs,imdata.flatten(),(new_x,new_y),method='cubic')

	return grid0

	# REBIN HIGH RES EPSF BACK TO ORIGINAL PIXEL SCALE 

def highbin_epsf_030(imdata):

	og_x,og_y = np.mgrid[0:99:100j, 0:99:100j]
	new_x,new_y = np.mgrid[0:99:300j,0:99:300j]
	XYpairs = np.dstack([og_x,og_y]).reshape(-1,2)
	grid0 = griddata(XYpairs,imdata.flatten(),(new_x,new_y),method='cubic')
	return grid0


def lowbin(ndarray, new_shape, operation='sum'):

    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# construct epsf using TARGETS in each image, to use the same portion of the detector:
# In loop:
# - Open file, use cutout to select the same region each time that will hopefully contain the target
# - Target will be brightest object in region, so use find peaks, sort by flux and just take brightest one
# (ie remove everything else from table?)
# - Then use extract stars to take out this target (use same parameters every time)
# - Append this star data to list (?), and then outside of loop feed into epsf builder?

# n needs to be a odd number
def construct_target_epsf(n):

	start=time.time()

	filenames = glob.glob('010_fits/*fits')

	#filenames = sorted(sys.argv[1:])
	datalist,starlist = [],[]
	for i in range(len(filenames)):
		imdataf = fits.open(str(filenames[i]))
		im = imdataf[1].data
		wcs = WCS(imdataf[0].header)

		position = [320, 280]
		size = [150, 150]
		cutout = Cutout2D(im, position=position, size=size, wcs=wcs)
		maxf = max(cutout.data.flatten())

		folder = filenames[i].split('/')[0]
		file = filenames[i].split('/')[1]

		plots = False
		if plots:
			plt.figure()
			plt.imshow(im,vmin=-0.2,vmax=2.0,origin='lowerleft')
			rect = plt.Rectangle(xy=(position[0]-size[0]/2,position[1]-size[0]/2),width=size[0],height=size[1],linewidth=1,edgecolor='r',fill=False)
			plt.gca().add_patch(rect)
			plt.savefig(str(folder)+'/cutouts/'+str(file)+'_fullfield.png')
			plt.close()

			plt.figure()
			plt.imshow(cutout.data,vmin=-0.2,vmax=2.0,origin='lowerleft')
			plt.savefig(str(folder)+'/cutouts/'+str(file)+'_cutout.png')
			plt.close()

		imdata = cutout.data
		#rebin imdata to higher resolution here
		imdata = highbin_epsf(imdata)
		#try using dao find - need to consider the centers/geometric shape of the objects to pick out center 
		simp = True
		if simp:
			peaks_tbl = find_peaks(imdata,threshold=maxf/2.)
		else:
			daofind = DAOStarFinder(threshold=maxf/2.,fwhm=15)
			peaks_tbl = daofind(imdata)

		stars_tbl = Table()
		if simp:
			stars_tbl['x'] = peaks_tbl['x_peak']
			stars_tbl['y'] = peaks_tbl['y_peak']
			stars_tbl['flux'] = peaks_tbl['peak_value']
		else:
			stars_tbl['x'] = peaks_tbl['xcentroid']
			stars_tbl['y'] = peaks_tbl['ycentroid']
			stars_tbl['flux'] = peaks_tbl['peak']

		stars_tbl.sort('flux')
		stars_tbl.reverse()
		keep_stars = Table(stars_tbl[0])
		keep_stars.remove_column('flux')

		# This is a simple way of background subtraction (just using mean). If background
		# varies across image, need to use a more complicated function -- check this!
		mean_val, median_val, std_val = sigma_clipped_stats(imdata,sigma=2.0)
		imdata -= median_val
		nddata = NDData(data=imdata)
		datalist.append(nddata)
		starlist.append(keep_stars)


	stars = extract_stars(datalist,starlist,size=35*10)
	#plot_extracted_stars(stars)

	# Documentation says that really maxiters should be ~10, but this takes a while..
	epsf_builder = EPSFBuilder(shape=(n*11,n*11),maxiters=10,progress_bar=True,center_accuracy=1)
	epsf,fitted_stars = epsf_builder(stars)

	#hdu = fits.PrimaryHDU(epsf.data)
	#hdul = fits.HDUList([hdu])
	#hdul.writeto('epsf_highbin_25.fits')
	
	plot_epsf(epsf)

	return epsf.data,stars
	#return stars

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# epsf subtraction using epsf cosntructed above using photutils
def do_epsf_subtraction(filename,starsn,imdata,imdata_vis,epsf,iraffind,daogroup,mmm_bkg):

	photometry_epsf = IterativelySubtractedPSFPhotometry(finder=iraffind,group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=epsf,fitter=LevMarLSQFitter(),niters=1,fitshape=(7,7),aperture_radius=7)
	result_tab = photometry_epsf(image=imdata)
	residual_image = photometry_epsf.get_residual_image()
	
	plt.figure()
	plt.imshow(residual_image,vmin=-0.2,vmax=2.0,cmap='plasma',origin='lowerleft')
	plt.show()

	hdu = fits.PrimaryHDU(residual_image)
	hdul = fits.HDUList([hdu])
	if 'als' in starsn:
		print('True')
		hdul.writeto(str(filename)+'_espf_residual_image_all.fits')
	else: 
		hdul.writeto(str(filename)+'_epsf_residual_image_1_iter_not_target_smaller_box.fits')

	return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# epsf subtraction using epsf constructed using tiny tim or PRF
def do_epsf_subtraction2(epsf,m,imdata,iraffind,daogroup,mmm_bkg,wcs,HIGHRES):

	#don't need to do subtraction for whole image here, as it takes ages for high res
	#read in epsf binned to x10 resolution: epsf
	#read in hubble image at observerd resolution: imdata, and bin to x10
	position = [320,280]
	size=[150,150]
	#position = [770,870]
	#size = [100, 100]

	cutoutdat = Cutout2D(imdata, position=position, size=size, wcs=wcs)
	cutoutim = cutoutdat.data
	if HIGHRES:
		cutoutim = highbin_epsf(cutoutim)

	if HIGHRES:
		fshape = m*11
		aprad = (m-1)*11
	else:
		fshape = m
		aprad = (m-1)
	#fshape=(m*6)-1
	#aprad = m*3

	# Function build for IR
	#photometry_epsf = IterativelySubtractedPSFPhotometry(finder=iraffind,group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=epsf,fitter=LevMarLSQFitter(),niters=1,fitshape=((m*11),(m*11)),aperture_radius=(m-1)*11)#multiply m by 11 for rebinning
	photometry_epsf = IterativelySubtractedPSFPhotometry(finder=iraffind,group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=epsf,fitter=LevMarLSQFitter(),niters=1,fitshape=(fshape,fshape),aperture_radius=aprad)

	result_tab = photometry_epsf(image=cutoutim)
	residual_image = photometry_epsf.get_residual_image()

	return residual_image,cutoutim

def manual_subtraction(center,epsf,n,sf):

	imfile = fits.open('../hst_data/serpens3/idxq28010_drz.fits')
	im = imfile[1].data
	wcs = WCS(imfile[0].header)

	im = highbin_im(im)	

	# NEED TO FIX THIS SO THAT STAR IS PROPERLY CENTERED IN CROP OF HIGH RES IMAGE!!

	fulllen = (n*11)+1
	x = int(fulllen/2)
	y = int(x-1)
	center = [a*10 for a in center]

	croptest = np.asarray(im[center[0]-x:center[0]+y,center[1]-x:center[1]+y],np.float64)

	#plt.figure()
	#plt.imshow(croptest,vmin=-0.2,vmax=2.0)
	#plt.show()

	maxi = np.nanmax(croptest.flatten())
	norm_im = (im/maxi)*256

	#epsf_high,m = construct_target_epsf(n)
	#epsf = lowbin(epsf_high,new_shape=(n,n),operation='mean')

	maxe = np.nanmax(epsf.flatten())
	norm_e = (epsf/maxe)*256

	norm_e = np.asarray([a*sf for a in norm_e],np.float64)
	plot_epsf(norm_e)

	crop = np.asarray(norm_im[center[0]-x:center[0]+y,center[1]-x:center[1]+y],np.float64)

	norm_im[center[0]-x:center[0]+y,center[1]-x:center[1]+y] = cv2.subtract(crop,norm_e)
	
	#plt.figure()
	#plt.imshow(norm_im[center[0]-2*x:center[0]+2*y,center[1]-2*x:center[1]+2*y],vmin=0.2,vmax=2.0,origin='lowerleft')
	#plt.savefig('test_subs/'+str(center[0])+'_'+str(center[1])+'_'+str(n)+'_'+str(sf)+'_serpens2epsf.png')
	#plt.close()
	#plt.show()

	return norm_im,crop


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PLANET INJECTION ALGORITM
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def planets(imdata,head,wcs,epsf,p,rads,npix,delta_m,HIGHRES):

	PLOTS = False
	# zoom in to target position and generate high resolution cut out
	position = [320,280]
	size = [150,150]
	cutoutdat = Cutout2D(imdata, position=position, size=size, wcs=wcs)
	cutoutim = cutoutdat.data
	if HIGHRES:
		cutoutim = highbin_epsf(cutoutim)
		s = 300
		pls = 45
		rad = 2
		fw = 15
	else:
		s = 30
		pls = 5
		rad = 2
		fw = 2
	targetcutout = Cutout2D(cutoutim,position=[len(cutoutim)/2,len(cutoutim)/2],size=s)
	targetim = targetcutout.data
	# simple background subtract - TRY REMOVING THIS LINE AS IT MIGHT BE MESSING WITH THE BACKGROUND SUBTRACTION LATER ON?
	#mean_val, median_val, std_val = sigma_clipped_stats(imdata,sigma=2.0)
	#targetim -= median_val

	# find magnitude of target using dao star finder
	maxf = max(targetim.flatten())
	find = DAOStarFinder(threshold=maxf*0.8,fwhm=fw)
	peaks_tbl = find(targetim)
	peaks_tbl.sort('peak')
	peaks_tbl.reverse()
	targetx = peaks_tbl[0]['xcentroid']
	targety = peaks_tbl[0]['ycentroid']

	#aperture photometry of target to check peak/flux values for noise
	
	aperture = CircularAperture([targetx,targety],r=rad)
	phot_table = aperture_photometry(targetim,aperture)
	area = np.pi*(rad**2.)

	f_ratio = 10**(delta_m/2.5)

	star_peak = phot_table[0]['aperture_sum']/area	
	planet_flux = phot_table[0]['aperture_sum']/f_ratio
	planet_peak = planet_flux/area
	# scale epsf to generate a planet signal
	maxe = max(epsf.flatten())
	epsfp = [(a/maxe)*planet_peak for a in epsf]
	center = int(len(epsfp)/2)

	# cuout just center of this array to isolate the planetary psf
	planetcutout =  Cutout2D(epsfp,position = [center,center],size = [pls,pls])
	pl_psf = planetcutout.data
	pl_psf_size = int(len(pl_psf)/2)

	#plt.figure()
	#plt.imshow(targetim,vmin=-0.5,vmax=2.0)
	#plt.show()

	#plt.figure()
	#plt.imshow(pl_psf,vmin=-0.5,vmax=2.0)
	#plt.show()

	# add this planet into the image data 
	# identify angle 0-360 to place target (0deg = 'north') and distance from center of target in arcsecs
	# HST WCF3 = 0.13 arcsec per pix
	# Start with 10pix (1.3arcsec) and 0deg (straight up) from target
	if HIGHRES:
		npix = npix*10.0 #as resolution is binned x10
	else:
		npix = npix
	center_planet = [int(targetx+npix*np.sin(rads)),int(targety-npix*np.cos(rads))]
	im_sec_to_add = targetim[center_planet[1]-(pl_psf_size):center_planet[1]+(pl_psf_size+1),center_planet[0]-(pl_psf_size):center_planet[0]+(pl_psf_size+1)]
	
	im_sec_to_add  = np.asarray(im_sec_to_add,np.float64)
	pl_psf = np.asarray(pl_psf,np.float64)	

	targetim[center_planet[1]-(pl_psf_size):center_planet[1]+(pl_psf_size+1),center_planet[0]-(pl_psf_size):center_planet[0]+(pl_psf_size+1)] = cv2.add(im_sec_to_add,pl_psf)

	if PLOTS:
		plt.figure()
		plt.imshow(targetim,origin='lowerleft',vmin=-0.5,vmax=5.0)
		plt.scatter([targetx,center_planet[0]],[targety,center_planet[1]],s=6,c='r',marker='x')
		#plt.xlim(100,350)
		#plt.ylim(150,400)
		#plt.savefig('planet_injection/32010/'+str(rads)+'_'+str(npix)+'_'+str(delta_m)+'_original.png')
		#plt.close()
		plt.show()

        # zoom in even more on target, remove all other stars
	#targetcutoutdat1 = Cutout2D(cutoutim,position=[targetx,targety],size=(p+5.)*10.)

	return targetim,[[targetx,center_planet[0]],[targety,center_planet[1]]],[targetx,targety]

def contrast_annuli(annulin,imdata,head,wcs,epsf,p,residim,HIGHRES):
	
	# Magnitude equation: m1 - m2 = 2.5*log10(f1/f2)
	# --> need FAINTER thing as object 1, BRIGHTER thing as object 2 to get a +VE delta mag
	# zoom in to target position and generate high resolution cut out
	position = [320,280]
	size = [150,150]
	cutoutdat = Cutout2D(imdata, position=position, size=size, wcs=wcs)
	cutoutim = cutoutdat.data
	if HIGHRES:
		cutoutim = highbin_epsf(cutoutim)
		s = 300
		rad = 20.
	else:
		s = 60
		rad = 2.
	targetcutout = Cutout2D(cutoutim,position=[len(cutoutim)/2,len(cutoutim)/2],size=s)
	targetim_unsub = targetcutout.data
	residimcutout = Cutout2D(residim,position=[len(cutoutim)/2,len(cutoutim)/2],size=s)
	targetim_sub = residimcutout.data
        # simple background subtract
	mean_val, median_val, std_val = sigma_clipped_stats(imdata,sigma=2.0)
	targetim_unsub -= median_val

        # find magnitude of target using dao star finder
	maxf = max(targetim_unsub.flatten())
	find = DAOStarFinder(threshold=10,fwhm=rad)
	peaks_tbl = find(targetim_unsub)
	peaks_tbl.sort('peak')
	peaks_tbl.reverse()
	targetx = peaks_tbl[0]['xcentroid']
	targety = peaks_tbl[0]['ycentroid']
	targetpeak = peaks_tbl[0]['peak']

	#annulus_aperture = CircularAnnulus([targetx,targety], r_in=1, r_out=2)
	#plt.figure()
	#plt.imshow(targetim_sub,vmin=-0.5,vmax=5.0)
	#annulus_aperture.plot(color='r',linewidth=5)
	#plt.show()

	aperture = CircularAperture([targetx,targety],r=rad*2)
	phot_table = aperture_photometry(targetim_unsub,aperture)
	targetcounts = phot_table['aperture_sum']
	
	element_radius = 0.53E-6/4.84814E-6 #lam/D for HST WCF3 and F127M, in radians
	element_area = np.pi*(element_radius)**2

	# lay down annuli around target star at varying distances
	radii = np.arange(1,annulin+2,1)
	deltam,starting_radius = [],[]
	for i in range(len(radii)-1):
		starting_radius.append(radii[i]*0.13)
		if HIGHRES:
			ri = radii[i]*10
			ro = radii[i+1]*10
			annulus_aperture = CircularAnnulus([targetx,targety], r_in=ri, r_out=ro)
		else:
			ri = radii[i]
			ro = radii[i+1]
			annulus_aperture = CircularAnnulus([targetx,targety],r_in=radii[i],r_out=radii[i+1])
		phot_table = aperture_photometry(targetim_sub,annulus_aperture) 
		
		#plt.figure()	
		#plt.imshow(targetim_sub,vmin=-0.5,vmax=5.0,origin='lowerleft')
		#annulus_aperture.plot(color='r',linewidth=5)
		#plt.show()

		annulus_area = np.pi*(ro**2-ri**2)
		n = annulus_area/element_area
		annulus_mask = annulus_aperture.to_mask(method='center')
		annulus_data = annulus_mask.multiply(targetim_sub)
		mask = annulus_mask.data
		annulus_data_1d = np.array(annulus_data[mask > 0])
		# If any of the annulus values are negative, set these to zero as they've been oversubtracted, and a negative flux isn't physical?
		#annulus_data_1d[annulus_data_1d < 0.0] = 0.0
		counts = np.sum(annulus_data_1d)
		std1 = np.std(annulus_data_1d)
		correction = np.sqrt(1+(1/n)) # n = number of resolution elements in this annulus (-1?)
		std2 = std1*correction
		f1 = 5.*std1
		f2 = targetpeak
		deltam.append(-2.5*(np.log10(f1/f2)))
	print(deltam)
	r = [a*0.13 for a in radii[0:annulin]]
	#plt.figure()
	#plt.plot(r,deltam)
	#plt.axis([min(r),max(r),max(deltam),min(deltam)])
	#plt.xlabel('Seperation (arcsec)')
	#plt.ylabel('Contrast ($\Delta$mag)')
	#pfilenames=['all_files010/idxq32010_drz.fits','all_files010/idxq28010_drz.fits']
	#plt.show()
	
	return deltam,starting_radius

def contrast_planets(ims,radius,dmags,angle,imdata,targetpos,HIGHRES,bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean,epsf):

	residim=low_res_psf(ims,targetpos,bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean,epsf)

	# NOT SURE IF THIS IS THE CORRECT APPROACH, BUT A RADIUS OF 0.5 PIX MAKES AN APERTURE THAT JUST MEASURES THE PEAK VALUE
	ap_radius = 1

	# Loop throught the full range of delta mags for each radius:
	#    --> Identify the delta mag at which the S/N of the planet >= 5
	# This is the delta mag to be plotted against this radius for the contrast curve, exit this loop/save this specific value?
	if HIGHRES:
		scale = 10.
	else:
		scale = 1.

	# Do aperture photometry at position that planet was injected		
	# Retreive planet position from info in dictionary 
	planetpos = [int(targetpos['x_0'][0]+scale*radius*np.sin(angle)),int(targetpos['y_0'][0]-scale*radius*np.cos(angle))]
	aperture = CircularAperture(planetpos,r=ap_radius*scale)

	#plt.figure()
	#plt.imshow(residim,vmin=-0.5,vmax=10.0,origin='lowerleft')
	#aperture.plot(color='r',linewidth=2)
	#plt.text(planetpos[0]+5,planetpos[1],str(dmags),color='white')
	#plt.show()	

	planet_table = aperture_photometry(residim,aperture)
	#planet_peak = planet_table[0]['aperture_sum']/(np.pi*(ap_radius*scale)**2.)
	planet_peak = planet_table[0]['aperture_sum']

	# Do aperture photometry on random, star-free backg locations and average to get background level
	# Create sigma clipped mask to inform on where sources are - should now be doing this with original, full image - change in resolution hopefully won't matter?
	# 	--> Will need to make aperture smaller to account for difference though 
	trimmed_im = Cutout2D(imdata,position=(291,259),size=(438,502))
	trimmed = trimmed_im.data
	mask = make_source_mask(trimmed, nsigma=3, npixels=4, dilate_size=11)	
	# Pick and random position in image and check if that pixel is masked out (should probs check if any of the pixels in the aperture are masked but not sure how...)
	im_coords = [(i,j) for j in range(len(trimmed)) for i in range(len(trimmed[0]))]	
	backg_coords,not_included = [],[]
	while len(backg_coords) < 2000:
		rand_coord = im_coords[int(np.random.uniform(0,len(im_coords)-1))]
		#print('Random coordinates:',rand_coord)
		mask_val = mask[rand_coord[1]][rand_coord[0]]
		#print('Random pix mask:',mask_val)
		if not mask_val:
			# check surrounding pixels to see if they are also not masked
			zoom_im_coords=[(i,j) for j in [rand_coord[1]-1,rand_coord[1],rand_coord[1]+1] for i in [rand_coord[0]-1,rand_coord[0],rand_coord[0]+1]]
			masks = []
			xzoomc,yzoomc = zip(*zoom_im_coords)
			if -1 in xzoomc or 502 in xzoomc or -1 in yzoomc or 438 in yzoomc or np.nan in xzoomc or np.nan in yzoomc:
		#		print('Edge case removed')
				not_included.append(rand_coord)
			else:	
				for i in range(len(zoom_im_coords)):
					mask_val = mask[zoom_im_coords[i][1]][zoom_im_coords[i][0]]
					masks.append(str(mask_val))		
			#		print('Surrounding mask values:',masks)
				if 'True' in masks:
					not_included.append(rand_coord)
				else:				
					backg_coords.append(rand_coord)
	aperture = CircularAperture(backg_coords,r=ap_radius)

	phot_table = aperture_photometry(trimmed,aperture)
	sumcol=np.asarray(phot_table['aperture_sum'])
	area = np.pi*ap_radius**2
	ap_value = []
	for k in range(len(sumcol)):
		b = sumcol[k]/area
		ap_value.append(b)
	mean_bkg = np.nanmean(ap_value)
	std_bkg = np.nanstd(ap_value)

	# Calculate S/N ratio using planet photometry and mean background level
	sn = (planet_peak-mean_bkg)/mean_bkg

	return sn,dmags


def low_res_psf(cutoutim,targetpos,bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean,epsf):
	photometry_epsf = []
	photometry_epsf = IterativelySubtractedPSFPhotometry(finder=iraffind,group_maker=daogroup,bkg_estimator=mmm_bkg,psf_model=epsf,fitter=LevMarLSQFitter(),niters=1,fitshape=((21),(21)),aperture_radius=22) 

	#plt.figure()
	#plt.imshow(cutoutim,vmin=-0.5,vmax=5.0,origin='lowerleft')
	#plt.show()

	result_tab = photometry_epsf(image=cutoutim,init_guesses=targetpos)
	residual_image = photometry_epsf.get_residual_image()

	return residual_image
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MAIN MAIN MAIN MAIN
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -:

p = int(sys.argv[1])
#p = 23

filenames = glob.glob('serp_010/*.fits')
#filenames = 'all_files030/idxq28030_drc.fits','all_files030/idxq31030_drc.fits'
annulin = 15
#filenames=['all_files010/idxq32010_drz.fits']

angle = 0.0
#angle = [(np.pi/4.0)*3]
radius = [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
#radius = [4.0,5.0,6.0,7.0]
arcsec_radius = [a*0.13 for a in radius]
#dmags = [8.0]
dmags = np.arange(2.5,8.0,0.25)
#dmags = np.arange(3.0,6.0,0.5)
dmags = dmags.tolist()
dmags.reverse()

PLANETS = True
CONTRAST = False
PLOTS = False
HIGHRES = False

#create a dictionary to store
if PLANETS:
	contrast1 = {'filename':'','deltam':[0 for i in range(len(radius))]}
	contrasts = [copy.deepcopy(contrast1) for i in range(len(filenames))]
else: 
	contrast1 = {'filename':'','deltam':[0 for i in range(annulin)]}
	contrasts = [copy.deepcopy(contrast1) for i in range(len(filenames))]

for i in range(len(filenames)):
	print(filenames[i])
	contrasts[i]['filename'] = filenames[i]
	imdataf = fits.open(str(filenames[i]))
	head = imdataf[0].header
	wcs = WCS(head)

	imdata,imdata_vis, bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean = init_setup(filenames[i])
	imdata_og,imdata_vis, bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean = init_setup(filenames[i])

	if HIGHRES:
		epsf_file = fits.open('epsf_highbin_'+str(p)+'.fits')
	else:
		epsf_file = fits.open('epsf_010_'+str(p)+'.fits')
	epsfs = epsf_file[0].data
	epsf = FittableImageModel(epsfs)

	#rads = np.arange(0,2*np.pi+2*np.pi/10.,2*np.pi/10.)
	#radius = [6.0,7.0]
	#npix = np.arange(1.,annulin+1,1)
	#dmags = [4.0,4.1,4.2,4.3,4.4,4.5]

	pl_resid_images = []
	if PLANETS:
		plot_value_mags = []
		for v in range(len(radius)):
			print('RADIUS  = ',radius[v])
			sns,deltamags = [],[]
			imdata,imdata_vis, bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean = init_setup(filenames[i])
			for w in range(len(dmags)):
	#			print(dmags[w])
				cutoutim = []
				cutoutim,planetpos,targetpos = planets(imdata,head,wcs,epsfs,p,angle,radius[v],dmags[w],HIGHRES)
				targettab = Table()
				x = Column([targetpos[0]],name='x_0')
				y = Column([targetpos[1]],name='y_0')
				targettab.add_columns([x,y])

				sn,deltamag = contrast_planets(cutoutim,radius[v],dmags[w],angle,imdata_og,targettab,HIGHRES,bkgrms, std, sigma_psf, iraffind, daogroup, mmm_bkg, mean,epsf) 	
				sns.append(sn)
				deltamags.append(deltamag)
			print(deltamags,sns)	
			plot_value_mags.append(deltamags[np.where(np.greater_equal(np.asarray(sns),5.0))[0][0]])
		contrasts[i]['deltam']=plot_value_mags
	else:
		residual_image,cutoutim = do_epsf_subtraction2(epsf,p,imdata,iraffind,daogroup,mmm_bkg,wcs,HIGHRES)

	if CONTRAST:
		c1,r = contrast_annuli(annulin,imdata,head,wcs,epsfs,p,residual_image,HIGHRES)
		contrasts[i]['deltam'] = c1
	if PLOTS:	
		fig, (ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True)
		ax1.imshow(cutoutim,origin='lowerleft',vmin=-0.02,vmax=0.05)
		ax2.imshow(residual_image,origin='lowerleft',vmin=-0.02,vmax=0.05)
		ax2.text(len(residual_image)/2+50,len(residual_image)/2,str(filenames[i].split('/')[1]),color='white')
		ax1.grid(False)
		ax2.grid(False)
		plt.savefig('residual_images/'+str(filenames[i].split('/')[1])+'_residual.png')
		plt.show()
		plt.close()

		hdu = fits.PrimaryHDU(residual_image)
		hdul = fits.HDUList([hdu])		
		hdul.writeto('residual_fits/resid'+str(filenames[i].split('/')[1]),overwrite=True)
if PLANETS:
	r = arcsec_radius
if CONTRAST or PLANETS:
	labels=['SS182949','SS182917','SS183032','SS182918','SS182953']
	cols = ['navy','darkorchid','magenta','orange','gold']
	# calculate mean contrast curve
	mean_contrast = []
	std_contrast_p,std_contrast_n=[],[]
	for i in range(len(contrasts[0]['deltam'])):
		mean_array=[]
		for j in range(len(contrasts)):
			mean_array.append(contrasts[j]['deltam'][i])
		mean_contrast.append(np.mean(mean_array))
		std_contrast_p.append(np.mean(mean_array)+np.std(mean_array))
		std_contrast_n.append(np.mean(mean_array)-np.std(mean_array))
	plt.figure()
	for i in range(len(contrasts)):
		plt.plot(r,contrasts[i]['deltam'],c=cols[i],label=labels[i],alpha=0.5)#,c='gray',linewidth=0.3)
	#plt.plot(r,mean_contrast,c='k',linewidth=2.)
	#plt.plot(r,std_contrast_p,c='k',linewidth=2.,linestyle='dashed')
	#plt.plot(r,std_contrast_n,c='k',linewidth=2.,linestyle='dashed')
	plt.axis([0,np.nanmax(r),np.nanmax(contrasts[0]['deltam'])+1,np.nanmin(contrasts[0]['deltam'])-1])	
	plt.axis([0,max(r),7,2])
	plt.vlines(0.13,np.nanmax(contrasts[0]['deltam'])+3,np.nanmin(contrasts[0]['deltam'])-3,color='gray',linestyle='dashed')
	plt.xlabel('Seperation (arcsec)')
	plt.ylabel('Contrast ($\Delta$mag)')
	plt.legend()
	plt.show()

