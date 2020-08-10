import sys
import pandas as pd
import rpy2
from rpy2.robjects.packages import importr
from pymer4 import Lmer
psych = importr('psych')
import numpy as np

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

import os
#import glob
#import seaborn as sns
from nilearn.connectome import ConnectivityMeasure
# from sklearn.linear_model import LinearRegression





# def lr_ifc(confounds_path,time_series_path,cfd_type):
#     if(time_series_path == ''):
#         print(f'timeseries {time_series_path} missing')
#         return None
#     cfd = pd.read_csv(confounds_path,sep='\t',index_col=0)
#     time_series = pd.read_csv(time_series_path,header=None).values
#     cols = [cc for cc in cfd.columns if 'censor' in cc]
#     cols.extend([cc for cc in cfd.columns if 'cosine' in cc])
#     if cfd_type != 'cc':
#         cols.extend([cc for cc in cfd.columns if 'trans' in cc])
#         cols.extend([cc for cc in cfd.columns if 'rot' in cc])
#     if cfd_type == 'ccmp' or cfd_type == 'ccmtp':
#         cols.extend([cc for cc in cfd.columns if 's0.' in cc])
#     if cfd_type == 'ccmt' or cfd_type == 'ccmtp':
#         cols.extend([cc for cc in cfd.columns if 'comp_cor' in cc])
#     censored_time_series = time_series.copy()
#     for yi in range(time_series.shape[1]):
#         X= cfd.loc[4:len(time_series)+3, cols[8:]]
#         y = time_series[:,yi]
#         lr = LinearRegression()
#         fitted = lr.fit(X,y)
#         pred = fitted.predict(X)
#         censored_time_series[:,yi] = y-pred
        

#     return censored_time_series

# def lr_bg(confounds_path,time_series_path,cfd_type):
#     if(time_series_path == ''):
#         print(f'timeseries {time_series_path} missing')
#         return None
#     cfd = pd.read_csv(confounds_path,sep='\t',index_col=0)
#     time_series = pd.read_csv(time_series_path,header=None).values
#     cols = [cc for cc in cfd.columns if 'censor' in cc]
#     cols.extend([cc for cc in cfd.columns if 'cosine' in cc])
#     if cfd_type != 'cc':
#         cols.extend([cc for cc in cfd.columns if 'trans' in cc])
#         cols.extend([cc for cc in cfd.columns if 'rot' in cc])
#     if cfd_type == 'ccmp' or cfd_type == 'ccmtp':
#         cols.extend([cc for cc in cfd.columns if 's0.' in cc])
#     if cfd_type == 'ccmt' or cfd_type == 'ccmtp':
#         cols.extend([cc for cc in cfd.columns if 'comp_cor' in cc])
#     censored_time_series = time_series.copy()
#     for yi in range(time_series.shape[1]):
#         X= cfd.loc[4:len(time_series)+3, cols]
#         y = time_series[:,yi]
#         lr = LinearRegression()
#         fitted = lr.fit(X,y)
#         pred = fitted.predict(X)
#         censored_time_series[:,yi] = y-pred

#     return censored_time_series

# def lr_rest(confounds_path,time_series_path,cfd_type):
#     if(time_series_path == ''):
#         print(f'timeseries {time_series_path} missing')
#         return None
#     cfd = pd.read_csv(confounds_path,sep='\t',index_col=0)
#     time_series = pd.read_csv(time_series_path,header=None).values
#     cols = [cc for cc in cfd.columns if 'censor' in cc]
#     cols.extend([cc for cc in cfd.columns if 'cosine' in cc])
#     if cfd_type != 'cc':
#         cols.extend([cc for cc in cfd.columns if 'trans' in cc])
#         cols.extend([cc for cc in cfd.columns if 'rot' in cc])
#     if cfd_type == 'ccmp' or cfd_type == 'ccmtp':
#         cols.extend([cc for cc in cfd.columns if 's0.' in cc])
#     if cfd_type == 'ccmt' or cfd_type == 'ccmtp':
#         cols.extend([cc for cc in cfd.columns if 'comp_cor' in cc])
#     censored_time_series = time_series.copy()
#     for yi in range(time_series.shape[1]):
#         X= cfd.loc[4:len(time_series)+3, cols]
#         y = time_series[:,yi]
#         lr = LinearRegression()
#         fitted = lr.fit(X,y)
#         pred = fitted.predict(X)
#         censored_time_series[:,yi] = y-pred
#     return censored_time_series




# def lr_concat(row,cfd_type):
#     ts = pd.read_csv(row.task_ts_path,header=None)
#     rs = pd.read_csv(row.rest_ts_path,header=None)
#     time_series = pd.concat([ts,rs],axis=0,join='inner',sort=False,ignore_index=True).values
#     if cfd_type == 'none':
#         return time_series
#     cfd_ts = pd.read_csv(row.cfds_physio_path_x, sep='\t')
#     cfd_rs = pd.read_csv(row.cfds_physio_path_y, sep='\t')
#     cfd = pd.concat([cfd_ts.loc[4:len(ts)+3],cfd_rs.loc[4:len(rs)+3]],axis=0,ignore_index=True,join='inner',sort=False)
#     cols = [cc for cc in cfd.columns if 'censor' in cc]
#     cols.extend([cc for cc in cfd.columns if 'cosine' in cc])
#     if cfd_type != 'cc':
#         cols.extend([cc for cc in cfd.columns if 'trans' in cc])
#         cols.extend([cc for cc in cfd.columns if 'rot' in cc])
#     if cfd_type == 'ccmp' or cfd_type == 'ccmtp':
#         cols.extend([cc for cc in cfd.columns if 's0.' in cc])
#     if cfd_type == 'ccmt' or cfd_type == 'ccmtp':
#         cols.extend([cc for cc in cfd.columns if 'comp_cor' in cc])
#     censored_time_series = time_series.copy()
#     for yi in range(time_series.shape[1]):
#         X= cfd.loc[:, cols]
#         y = time_series[:,yi]
#         lr = LinearRegression()
#         fitted = lr.fit(X,y)
#         pred = fitted.predict(X)
#         censored_time_series[:,yi] = y-pred
#     return censored_time_series

def run_corr(metric,cfd,dat_paths,ts_dir,corr_output_dir,icc_output_dir,bs_col):
    if not os.path.isfile(f'{icc_output_dir}/rs_icc_{bs_col.name}_cfd_{cfd_alph}.csv'):
        print('not rs file')
        rs_ts = []
#         if metric == ['ifc']:
#             for _,row in dat_paths.iterrows():
#                 rs_ts.append(lr_ifc(row.cfds_physio_path_x,row.task_ts_path, cfd))
#         elif metric == ['bgc']:
#             for _,row in dat_paths.iterrows():
#                 rs_ts.append(lr_bg(row.cfds_physio_path_x,row.task_ts_path, cfd))
#         elif metric == ['rs']:
#             for _,row in dat_paths.iterrows():
#                 rs_ts.append(lr_rest(row.cfds_physio_path_y,row.rest_ts_path, cfd))
#         else:
#             for _,row in dat_paths.iterrows():
#                 rs_ts.append(lr_concat(row, cfd))
        for _,row in dat_paths.iterrows():
            fname = f'{ts_dir}/basc_ts_subject_{row.subject}_session_{int(row.session)}_run_{int(row.run)}_metric_{metric}_cfds_{cfd}.csv'
            rs_ts.append(pd.read_csv(fname).values)
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrices = correlation_measure.fit_transform(rs_ts)
        pd.DataFrame(correlation_measure.mean_).to_csv(f'{corr_output_dir}/{metric}_cm_{bs_col.name}_cfd_{cfd}.csv',header=None,index=None)
        return correlation_matrices
    else:
        if metric == 'concat':
            rs_ts = []
            for _,row in dat_paths.iterrows():
                rs_file = f'{ts_dir}/basc_ts_subject_{row.subject}_session_{row.session}_run_{row.run}_metric_rs_cfds_{cfd}.csv'
                bg_file = f'{ts_dir}/basc_ts_subject_{row.subject}_session_{row.session}_run_{row.run}_metric_bgc_cfds_{cfd}.csv'
                rs_ts.append(pd.read_csv(rs_file,header=None).append(pd.read_csv(bg_file,header=None),sort=False,ignore_index=True).values)
            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrices = correlation_measure.fit_transform(rs_ts)
            csv = f'{corr_output_dir}/{metric}_cm_{bs_col.name}_cfd_{cfd}.csv'
            pd.DataFrame(correlation_measure.mean_).to_csv(csv,header=None,index=None)
        if metric == 'concat_ifc':
            print('metric is concat_ifc')
            rs_ts = []
            for _,row in dat_paths.iterrows():
                rs_file = f'{ts_dir}/basc_ts_subject_{row.subject}_session_{row.session}_run_{row.run}_metric_rs_cfds_{cfd}.csv'
                bg_file = f'{ts_dir}/basc_ts_subject_{row.subject}_session_{row.session}_run_{row.run}_metric_ifc_cfds_{cfd}.csv'
                rs_ts.append(pd.read_csv(rs_file,header=None).append(pd.read_csv(bg_file,header=None),sort=False,ignore_index=True).values)
            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrices = correlation_measure.fit_transform(rs_ts)
            csv = f'{corr_output_dir}/{metric}_cm_{bs_col.name}_cfd_{cfd}.csv'
            pd.DataFrame(correlation_measure.mean_).to_csv(csv,header=None,index=None)

        return correlation_matrices
    return []


def icc(bs_col,corr_matrices,cfd_alph,dat_paths,metric,output_dir,alph=.003671):
#def icc(corr_matrices,cfd_alph,dat_paths,metric,icc_dir,alph=.003671):
    if (corr_matrices != []):
        icc_dat = pd.DataFrame()


        icc_dat['unique_subject'] = ['']
        icc_dat['session'] = ['']

        for i in range(corr_matrices.shape[1]):
            for j in range(corr_matrices.shape[2]):
                icc_dat['{}_{}'.format(i,j)] = [0.0]

        dat_paths['unique_subject'] = dat_paths['subject']
        while(len(dat_paths.loc[dat_paths.duplicated(subset=['unique_subject','session'])]) > 0):
            for _,row in dat_paths.loc[dat_paths.duplicated(subset=['unique_subject','session'])].iterrows():
                dat_paths['unique_subject'][_] = row.unique_subject+10000

        if metric == 'rs':
            task = 'rest'
        else:
            task = 'mid'
        for _,row in dat_paths.iterrows():
#             corr = pd.read_csv(f'{corr_dir}/{metric}_cm_{row.subject}_session_{row.session}_run_{row.run}_task_{task}_cfd_{cfd_alph}.csv',header=None)
            corr = corr_matrices[_]
            newrow = []
            newrow.append('{}'.format(row.unique_subject))
            newrow.append('{}'.format(row.session))
            for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        newrow.append(corr[i][j])
            icc_dat.loc[_] = newrow

        coldrop = []
        for x in range(2,len(icc_dat.columns)):
            meas = icc_dat.columns[x].split('_')
            if int(meas[0]) <= int(meas[1]):
                coldrop.append(icc_dat.columns[x])
        icc_dat = icc_dat.drop(columns=coldrop)
        measure_iccs=[]

        measure_cols = icc_dat.columns[2:]
        for mc in measure_cols:
            psych_dat = icc_dat.loc[:, ['unique_subject', 'session', mc]].set_index(['unique_subject', 'session']).unstack()
            mc_res = psych.ICC(psych_dat, missing=False,alpha=alph)
            mc_res = mc_res[0].loc[['Single_random_raters', 'Single_fixed_raters']]
            mc_res['measure'] = mc
            measure_iccs.append(mc_res)
        measure_iccs = pd.concat(measure_iccs)    

        #return pd.DataFrame(measure_iccs)
        measure_df = pd.DataFrame(measure_iccs)
        measure_df.to_csv(f'{output_dir}/{metric}_icc_{bs_col.name}_cfd_{cfd_alph}.csv')

if __name__ == "__main__": 
    bs_num = sys.argv[1]
    corr_output_dir = sys.argv[2]
    cfd_alph = sys.argv[3]
    dat_paths = sys.argv[4]
    ts_dir = sys.argv[5]
    metric = sys.argv[6]
    icc_output_dir = sys.argv[7]
    
    boots = pd.read_csv('/data/MBDU/midla/notebooks/campcc/mid_v2/test_retest/bootstrap.csv')
    bs_col = boots.loc[:, f'boot_{int(bs_num):05d}']
    print('testing')
    if not os.path.isfile(f'{corr_output_dir}/{metric}_cm_{bs_col.name}_cfd_{cfd_alph}.csv'):
        print('passed, running')
        dat_paths = pd.read_csv(dat_paths)
        dat_paths = pd.DataFrame(bs_col).rename(columns={bs_col.name:'subject'}).merge(dat_paths, how='left', on=['subject'])
        icc(bs_col,run_corr(metric,cfd_alph,dat_paths,ts_dir,corr_output_dir,icc_output_dir,bs_col),cfd_alph,dat_paths,metric,icc_output_dir)
