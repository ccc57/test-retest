import sys
import pandas as pd
import nilearn
import nibabel as nib
import nltools.prefs as prf
from nilearn import datasets, image, input_data
from nilearn.masking import apply_mask
from nilearn.image import resample_to_img
from nilearn.image import concat_imgs, index_img
import numpy as np


#Function: takes in row and masker object
#Returns image masked by grey matter, confounds, and transformed to atlas masker
def gm_weight_and_mask_task(row, regions_extracted_img, metric,cfd_type,n_dummy=4, min_trs=100,**kwargs):
    img = image.load_img(row.scaled_path_x)
    if img.get_fdata().shape[-1] < min_trs:
        return None
    # load subjects grey matter mask
    mask_img = resample_to_img(image.load_img(row.mask_path_x),img)
    mask_dat = mask_img.get_fdata()[:,:,:,np.newaxis]
    
    # make a binary grey matter mask
    mask_bin = mask_dat > 0
    
    # mask the regions image by the subject grey matter
    regions_masked_dat = regions_extracted_img.get_fdata() * mask_bin
    regions_masked_img = image.new_img_like(regions_extracted_img, 
                                            regions_masked_dat, 
                                            affine=regions_extracted_img.affine, 
                                            copy_header=True)
    
    # create a new masker instance
    masker = input_data.NiftiMapsMasker(
        regions_extracted_img, resampling_target="data", **kwargs)
    
    # apply the grey matter mask to the image data
    gm_dat = mask_dat * img.get_fdata()
    gm_img = image.new_img_like(img, gm_dat, affine=img.affine, copy_header=True)
    
    cfd = pd.read_csv(row.cfds_physio_path_x, sep='\t',index_col=0)
    cols = [cc for cc in cfd.columns if 'censor' in cc]
    cols.extend([cc for cc in cfd.columns if 'cosine' in cc])
    if cfd_type != 'cc':
        cols.extend([cc for cc in cfd.columns if 'trans' in cc])
        cols.extend([cc for cc in cfd.columns if 'rot' in cc])
    if cfd_type == 'ccmp' or cfd_type == 'ccmtp':
        cols.extend([cc for cc in cfd.columns if 's0.' in cc])
    if cfd_type == 'ccmt' or cfd_type == 'ccmtp':
        cols.extend([cc for cc in cfd.columns if 'comp_cor' in cc])
    if metric == 'ifc':
        cfds= cfd.loc[:, cols[8:]]
    else:
        cfds= cfd.loc[:, cols]
    print(f'/data/MBDU/midla/notebooks/campcc/subject_ts/basc_ts_subject_{row.subject}_session_{row.session}_run_{row.run}_metric_{metric}_cfds_{cfd_type}.csv')
    pd.DataFrame(masker.fit_transform(gm_img,cfds.values)[n_dummy:,:]).to_csv(f'/data/MBDU/midla/notebooks/campcc/subject_ts/basc_ts_subject_{row.subject}_session_{row.session}_run_{row.run}_metric_{metric}_cfds_{cfd_type}.csv',header=None, index=None)
        
    
#Function: takes in row and masker object
#Returns image masked by grey matter, confounds, and transformed to atlas masker
def gm_weight_and_mask_rest(row, regions_extracted_img,metric,cfd_type, n_dummy=4, min_trs=100, **kwargs):
    img = image.load_img(row.scaled_path_y)
    if img.get_fdata().shape[-1] < min_trs:
        return None
    # load subjects grey matter mask
    mask_img = resample_to_img(image.load_img(row.mask_path_y),img)
    mask_dat = mask_img.get_fdata()[:,:,:,np.newaxis]
    
    # make a binary grey matter mask
    mask_bin = mask_dat > 0
    
    # mask the regions image by the subject grey matter
    regions_masked_dat = regions_extracted_img.get_fdata() * mask_bin
    regions_masked_img = image.new_img_like(regions_extracted_img, 
                                            regions_masked_dat, 
                                            affine=regions_extracted_img.affine, 
                                            copy_header=True)
    
    # create a new masker instance
    masker = input_data.NiftiMapsMasker(
        regions_masked_img, resampling_target="data", **kwargs)
    
    # apply the grey matter mask to the image data
    gm_dat = mask_dat * img.get_fdata()
    gm_img = image.new_img_like(img, gm_dat, affine=img.affine, copy_header=True)
    
    
    cfd = pd.read_csv(row.cfds_physio_path_y, sep='\t',index_col=0)
    cols = [cc for cc in cfd.columns if 'censor' in cc]
    cols.extend([cc for cc in cfd.columns if 'cosine' in cc])
    if cfd_type != 'cc':
        cols.extend([cc for cc in cfd.columns if 'trans' in cc])
        cols.extend([cc for cc in cfd.columns if 'rot' in cc])
    if cfd_type == 'ccmp' or cfd_type == 'ccmtp':
        cols.extend([cc for cc in cfd.columns if 's0.' in cc])
    if cfd_type == 'ccmt' or cfd_type == 'ccmtp':
        cols.extend([cc for cc in cfd.columns if 'comp_cor' in cc])
    cfds= cfd.loc[:, cols]
    pd.DataFrame(masker.fit_transform(gm_img,cfds.values)[n_dummy:, :]).to_csv(f'/data/MBDU/midla/notebooks/campcc/subject_ts/basc_ts_subject_{row.subject}_session_{row.session}_run_{row.run}_metric_{metric}_cfds_{cfd_type}.csv',header=None, index=None)

    
    
if __name__ == "__main__": 
    dat_path = sys.argv[1]
    atlas_path = sys.argv[2]
    row = sys.argv[3]
    metric = sys.argv[4]
    cfd_type = sys.argv[5]
    
    dat_paths = pd.read_csv(dat_path,index_col=0)
    regions_extracted_img = image.load_img(atlas_path)
    print(f'{metric}_{cfd_type}_{row}')
    if metric in ['ifc','bgc']:
        row=dat_paths.iloc[int(row)]
        gm_weight_and_mask_task(row,regions_extracted_img,metric,cfd_type,t_r=2,detrend=True,low_pass=.1, high_pass=.01,standardize=False, memory='nilearn_cache', memory_level=1)
    else:
        row=dat_paths.iloc[int(row)]
        gm_weight_and_mask_rest(row,regions_extracted_img,metric,cfd_type,t_r=2.5,detrend=False,low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1)
   