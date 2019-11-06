import numpy as np
import rasterio
from skimage.morphology import black_tophat, square, white_tophat
from skimage.transform import rotate



def grayscale_raster_creation(input_MSfile, output_filename):
    """ 
    This function creates a grayscale brightness image from an input image to be used for MBI calculation. For every pixel 
    in the input image, the intensity values from the red, green, blue channels are first obtained, and the maximum of these values 
    are then assigned as the pixel's intensity value, which would give the grayscale brightness image as mentioned earlier, as per 
    standard practice in the remote sensing academia and industry. It is assumed that the first three channels of the input image 
    correspond to the red, green and blue channels, irrespective of order.
    
    Inputs:
    - input_MSfile: File path of the input image that needs to be converted to grayscale brightness image
    - output_filename: File path of the grayscale brightness image that is to be written to file
    
    Outputs:
    - gray: Numpy array of grayscale brightness image of corresponding multi - channel input image
    
    """
    
    with rasterio.open(input_MSfile) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])[:, :, 0 : 3]
    
    gray = np.max(img, axis = 2).astype(metadata['dtype'])
    
    metadata['count'] = 1
    
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        dst.write(gray[np.newaxis, :, :])
    
    return gray

  
  
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
    
    with rasterio.open(input_grayfile) as f:
        metadata = f.profile
        gray = f.read(1)
    
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

    MBI = (np.sum(DMP_MBI_list, axis = 0) / (4 * (((s_max - s_min) / delta_s) + 1))).astype(np.float32)
    
    if calc_MSI:
        MSI = (np.sum(DMP_MSI_list, axis = 0) / (4 * (((s_max - s_min) / delta_s) + 1))).astype(np.float32)
    
    metadata['dtype'] = 'float32'
    if write_MBI:
        with rasterio.open(output_MBIname, 'w', **metadata) as mbi_dst:
            mbi_dst.write(MBI[np.newaxis, :, :])
    
    if write_MSI:
        with rasterio.open(output_MSIname, 'w', **metadata) as msi_dst:
            msi_dst.write(MSI[np.newaxis, :, :])
    
    if calc_MSI:    
        return MBI, MSI
    else:
        return MBI
