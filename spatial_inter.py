# spatial_interpolation
# Author: Miranda Lv
# Date: 1/27/2015
# Note: This code is used to do spatial interpretation for DET


import sklearn
import gdal
import osr
from sklearn.gaussian_process import GaussianProcess
import numpy
from matplotlib import pyplot as pl
import matplotlib.image as mpimg

"""
opts, args= getopt.getopt(sys.argv[1:], "ho:v", ["help", "output="]) #Need to polish the inputs later
inf = args[0]
outf = args[1]
num = args[2]
"""
inf = r"/home/mirandalv/Downloads/del.tif" 
outf = r"/home/mirandalv/Downloads/del_out.tif"

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.ReadAsArray()

#----
# Create a new array canvas for the output

def nanarray(old_r, old_c, num):
	new_r = old_r * num
	new_c = old_c * num
	newarray = numpy.empty((new_r, new_c,))
	newarray[:] = numpy.nan
	return newarray

def newarray(old_array, new_array, num):
	new_r = new_array.shape[0]
	new_c = new_array.shape[1]
	old_r = old_array.shape[0]
	old_c = old_array.shape[1]
	start_r = (num - 1)/2 +1
	#end_r = new_r - (start)
	for i in range(0, old_r):
		start_c = (num - 1)/2 +1
		for j in range(0, old_c):
			new_array[start_r][start_c] = old_array[i][j]
			start_c = start_c + num
		start_r += num
	return new_array



def originrst(rasterfn):
	rstdic = {}
	raster = gdal.Open(rasterfn)
	geotransform = raster.GetGeoTransform()
	originX = geotransform[0]
	originY = geotransform[3]
	pixelWidth = geotransform[1]
	pixelHeight = geotransform[5]
	rstdic["originX"] = originX
	rstdic["originY"] = originY
	rstdic["pxWidth"] = pixelWidth
	rstdic["pxHeight"] = pixelHeight
	return rstdic


def array2raster(newRasterfn,oldrstdic,inarray,num): 
	# num is how many rows/cols the original pixel size divided by
	# We can make num as an constant number
    cols = inarray.shape[1]
    rows = inarray.shape[0]
    originX = oldrstdic["originX"]
    originY = oldrstdic["originY"]
    pixelWidth = oldrstdic["pxWidth"]/num
    pixelHeight = oldrstdic["pxHeight"]/num
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(inarray)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

"""
def array2raster(rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
"""


in_array = raster2array(inf)
in_rnum = in_array.shape[0]
in_cnum = in_array.shape[1]
new_nanarray = nanarray(in_rnum, in_cnum, 11)
out_array = newarray(in_array, new_nanarray, 11)

"""
print in_rnum, in_cnum
print out_array.shape[0], out_array.shape[1]
"""


r = numpy.linspace(0, 1, out_array.shape[1])
c = numpy.linspace(0, 1, out_array.shape[0])

rr, cc = numpy.meshgrid(r, c)
vals = ~numpy.isnan(out_array)


#---
# Gaussian Process Regression
gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.01)
gp.fit(X=numpy.column_stack([rr[vals],cc[vals]]), y=out_array[vals])
rr_cc_as_cols = numpy.column_stack([rr.flatten(), cc.flatten()])
interpolated = gp.predict(rr_cc_as_cols).reshape(out_array.shape)

oldrst = originrst(inf)
array2raster(outf,oldrst,interpolated,11)


#imgplot_out = pl.imshow(interpolated)
#imgplot_in = pl.imshow(in_array)
#imgplot_more = pl.imshow(out_array)
#pl.show()





#---
# raster to center point shapefile


#---
# point shapefile using kriging to predict other locations



#---
# raster t


