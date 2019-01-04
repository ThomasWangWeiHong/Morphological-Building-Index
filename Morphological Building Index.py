import numpy as np
from osgeo import gdal
from skimage.morphology import black_tophat, square, white_tophat
from skimage.transform import rotate



def MBI_MSI_calculation_and_feature_map_creation(input_grayfile, output_MBIname, output_MSIname, s_min, s_max, delta_s, 
                                                 calc_MSI = False, write_MBI = True, write_MSI = False):
    """ 
    This function is used to calculate the Morphological Building Index (MBI) as proposed in the paper 'Morphological 
    Building  - Shadow Index for Building Extraction From High - Resolution Imagery Over Urban Areas' by Xin Huang and 
    Liangpei Zhang (2012). 
    
    Inputs:
    - input_grayfile: String or path of grayscale tif file to be used.
    - output_MBIname: String or path of MBI feature map to be written.
    - output_MSIname: String or path of MSI feature map to be written.
    - s_min: Minimum scale size to be used. (must be greater than or equal to 3).
    - s_max: Maximum scale size to be used.
    - delta_s: Spatial increment for scale size
    - calc_MS: Boolean indicating whether to calculate MSI.
    - write_MBI: Boolean indicating whether to write MBI feature map to file.
    - write_MSI: Boolean indicating whether to write MSI feature map to file.
    
    Outputs:
    - MBI: MBI feature map for input grayscale image.
    - MSI (optional): MSI feature map for input grayscale image.
    
    """
        
    if s_min < 3:
        raise ValueError('s_min must be greater than or equal to 3.')
    
    gray = gdal.Open(input_grayfile).ReadAsArray()    
    MP_MBI_list = []
    MP_MSI_list = []
    DMP_MBI_list = []
    DMP_MSI_list = []
    
    for i in range(s_min, s_max + 1, 2 * delta_s):
        SE_intermediate = square(i)
        SE_intermediate[ : int((i - 1) / 2), :] = 0
        SE_intermediate[int(((i - 1) / 2) + 1) : , :] = 0
        
        SE_1 = SE_intermediate
        SE_2 = rotate(SE_1, 45, order = 0, preserve_range = True).astype('uint8')
        SE_3 = rotate(SE_1, 90, order = 0, preserve_range = True).astype('uint8')
        SE_4 = rotate(SE_1, 135, order = 0, preserve_range = True).astype('uint8')
        
        MP_MBI_1 = white_tophat(gray, selem = SE_1)
        MP_MBI_2 = white_tophat(gray, selem = SE_2)
        MP_MBI_3 = white_tophat(gray, selem = SE_3)
        MP_MBI_4 = white_tophat(gray, selem = SE_4)
        
        if calc_MSI: 
            MP_MSI_1 = black_tophat(gray, selem = SE_1)
            MP_MSI_2 = black_tophat(gray, selem = SE_2)
            MP_MSI_3 = black_tophat(gray, selem = SE_3)
            MP_MSI_4 = black_tophat(gray, selem = SE_4)
            
            MP_MSI_list.append(MP_MSI_1)
            MP_MSI_list.append(MP_MSI_2)
            MP_MSI_list.append(MP_MSI_3)
            MP_MSI_list.append(MP_MSI_4)
        
        MP_MBI_list.append(MP_MBI_1)
        MP_MBI_list.append(MP_MBI_2)
        MP_MBI_list.append(MP_MBI_3)
        MP_MBI_list.append(MP_MBI_4)
        
    
    for j in range(4, len(MP_MBI_list), 1):
        DMP_MBI_1 = np.absolute(MP_MBI_list[j] - MP_MBI_list[j - 4])
        DMP_MBI_list.append(DMP_MBI_1)
                
        if calc_MSI:
            DMP_MSI_1 = np.absolute(MP_MSI_list[j] - MP_MSI_list[j - 4])
            DMP_MSI_list.append(DMP_MSI_1)

    MBI = np.sum(DMP_MBI_list, axis = 0) / (4 * (((s_max - s_min) / delta_s) + 1))
    
    if calc_MSI:
        MSI = np.sum(DMP_MSI_list, axis = 0) / (4 * (((s_max - s_min) / delta_s) + 1))
    
    if write_MBI:
        input_dataset = gdal.Open(input_grayfile)
        input_band = input_dataset.GetRasterBand(1)
        gtiff_driver = gdal.GetDriverByName('GTiff')
        output_dataset = gtiff_driver.Create(output_MBIname, input_band.XSize, input_band.YSize, 1, gdal.GDT_Float32)
        output_dataset.SetProjection(input_dataset.GetProjection())
        output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
        output_dataset.GetRasterBand(1).WriteArray(MBI)
    
        output_dataset.FlushCache()
        del output_dataset
    
    if write_MSI:
        input_dataset_2 = gdal.Open(input_grayfile)
        input_band_2 = input_dataset_2.GetRasterBand(1)
        gtiff_driver_2 = gdal.GetDriverByName('GTiff')
        output_dataset_2 = gtiff_driver_2.Create(output_MSIname, input_band_2.XSize, input_band_2.YSize, 1, gdal.GDT_Float32)
        output_dataset_2.SetProjection(input_dataset_2.GetProjection())
        output_dataset_2.SetGeoTransform(input_dataset_2.GetGeoTransform())
        output_dataset_2.GetRasterBand(1).WriteArray(MSI)
    
        output_dataset_2.FlushCache()
        del output_dataset_2
    
    if calc_MSI:    
        return MBI, MSI
    else:
        return MBI