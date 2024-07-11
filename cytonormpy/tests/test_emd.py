import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import os
import fnmatch
import readfcs
import re

def calculate_emds(input_directory, files, channels,input_directory_ct=None,ct_files=None,cell_types_list=None,transform=False):
    '''
    Input:
    - input_directory (str) : directory where the fcs files are stored
    - files (list) : list of fcs files
    - channels (list) : list of channels to be used for the analysis
    - input_directory_ct (str) : directory where the csv files containing cell type information are stored
    - ct_files (list) : list of csv files containing cell type information
    - cell_types_list (list) : list of cell types to be included in the analysis
    - transform_data = False (bool) : whether to apply arcsinh(value/5) transformation to the data

    Returns:
    > If cell information are provided: a dict in the form of {channel1: {cell_type1: emd, cell_type2: emd,...}, channel2: {cell_type1: emd,cell_type2: emd,...},...}
    > If cell information are not provided: a dict in the form of {channel1: emd, channel2: emd,...}

    Note:
    > The function assumes that the order of files in the list 'files' is the same as the order of files in the list 'ct_files'
    '''  
    dict_channels_ct= create_marker_dictionary_ct(input_directory,files,channels,input_directory_ct,ct_files,cell_types_list,transform_data=transform)
    emds_dict= compute_emds_fromdict_ct(dict_channels_ct,cell_types_list = cell_types_list,num_batches=len(files))
    return emds_dict

def create_marker_dictionary_ct(input_directory,files,channels,input_directory_ct,ct_files,cell_types_list,transform_data=False):
    '''
    Input: 
    - input_directory (str) : directory where the fcs files are stored
    - files (list) : list of fcs files
    - channels (list) : list of channels to be used for the analysis
    - input_directory_ct (str) : directory where the csv files containing cell type information are stored
    - cell_types_list (list) : list of cell types to be included in the analysis
    - ct_files (list) : list of csv files containing cell type information 
    - transform = False (bool) : whether to apply arcsinh(value/5) transformation to the data

    Returns: 
    > If cell information are provided: a dict in the form of {channel1: {cell_type1: [[batch1],[batch2],...,[batch10]], cell_type2: [[batch1],[batch2],...}, channel2: {cell_type1: [[batch1],[batch2],...,],...}...}
    > If cell information are not provided: a dict in the form of {channel1: [[batch1],[batch2],...,[batch10]], channel2: [[batch1],[batch2],...],...}

    Note:
    > The function assumes that the order of files in the list 'files' is the same as the order of files in the list 'ct_files'
    
    '''
    channels_dict={} 
    # initialize the dictionary
    channels_dict = {c: {} for c in channels}
    #Iterate over files
    num_batches = len(files)

    
    for i in range(num_batches):
        fcs = files[i]
        adata= readfcs.read(input_directory+fcs) #create anndata object from fcs file
        df = adata.to_df()
        df.columns= list(adata.var['channel'])

        if cell_types_list:
            ct_file = ct_files[i]    
            ct_annotations = pd.read_csv(input_directory_ct+ct_file)       
            ct_annotations =  list(ct_annotations.iloc[:,0])
            df['cell_type'] = ct_annotations 

        if cell_types_list != None:
            # Compute dictionary for each cell type
            for c in channels:
                df_channel_ct = df.loc[:,['cell_type',c]]
                for ct in cell_types_list:
                    marker_array= df_channel_ct[df_channel_ct['cell_type']==ct]
                    marker_array= marker_array[c].values
                    if transform_data == True:
                        marker_array= np.arcsinh(marker_array/5)
                    else:
                        pass
                    ct_label = ct.replace(' ','_')

                    if ct_label not in channels_dict[c].keys():         # If dictionary is empty, initialize the dictionary with the cell type label
                        channels_dict[c][ct_label] = []
                    
                    channels_dict[c][ct_label].append(marker_array)

        for c in channels:
            marker_array = df.loc[:,c].values          
            if transform_data == True:
                marker_array = np.arcsinh(marker_array/5)
            else:
                pass
            
            if "All_cells" not in channels_dict[c].keys():      # If dictionary is empty, initialize the dictionary with the 'all_cells' label
                channels_dict[c]["All_cells"] = []
           
            channels_dict[c]["All_cells"].append(marker_array)

   
                    

    return channels_dict    



def compute_emds_fromdict_ct(channels_dict,cell_types_list,num_batches):
    '''
    Input:
    - channels_dict (dict) : dictionary computed using 'create_marker_dictionary_ct' function
    - cell_types_list (list) : list of cell types to be included in the analysis
    - num_batches (int) : number of batches

    Returns:
    > a dictionary in the form of {channel1: {cell_type1: emd, channel2: emd, ...}, channel2: {cell_type1: emd,cell_type2: emd},...}

    '''

    emds_dict = {}

    # Initialize dict
    for c in channels_dict.keys():
        emds_dict[c] = {}
        if cell_types_list != None:
            for ct in cell_types_list:
                ct_label = ct.replace(' ','_')
                emds_dict[c][ct_label]=0
                
                #compute pairwise EMDs among batches for the channel c, cell type ct
                for i in range(num_batches):
                    for j in range(i+1,num_batches):  
                        #emd= wasserstein_distance(channels_dict[c][ct_label][i],channels_dict[c][ct_label][j])
                        u_values, u_weights = bin_array(channels_dict[c][ct_label][i])
                        v_values, v_weights = bin_array(channels_dict[c][ct_label][j])
                        emd = wasserstein_distance(u_values, v_values, u_weights, v_weights)
                        if emd > emds_dict[c][ct_label]:
                            emds_dict[c][ct_label]=emd   

    for c in channels_dict.keys():
        emds_dict[c]["All_cells"]=0 
        for i in range(num_batches):
            for j in range(i+1,num_batches):
                u_values, u_weights = bin_array(channels_dict[c]["All_cells"][i])
                v_values, v_weights = bin_array(channels_dict[c]["All_cells"][j])
                emd = wasserstein_distance(u_values, v_values, u_weights, v_weights)
                if emd > emds_dict[c]["All_cells"]:
                    emds_dict[c]["All_cells"]=emd
                        
    return emds_dict

def bin_array(values):
    ''''
    Input:
    - values (array) : array of values

    eeturns:
    > a tuple with two arrays: the first array contains the binning, the second array contains the bin weights used to compute the EMD in the 'compute_emds_fromdict_ct' function
    '''
    bins = np.arange(-100, 100.1, 0.1)+0.0000001  # 2000 bins, the 0.0000001 is to avoid the left edge being included in the bin (Mainly impacting 0 values)
    counts, _ = np.histogram(values, bins=bins)
    
    return range(0,2000), counts/sum(counts)


def wrap_results(distances_before,distances_after):
    ''''
     Input:
     - distances_before (dict) : dictionary of EMDs before normalization. Computed using 'calculate_emds' function
     - distances_after (dict) : dictionary of EMDs after normalization. Computed using 'calculate_emds' function

     Returns:
     > a pd.DataFrame with the following columns: 'cell_type', 'channel', 'emd_before', 'emd_after'
     '''
    df1 = pd.DataFrame(distances_before)
    df1['cell_type'] = df1.index
    df1 = df1.melt("cell_type")
            
    df2 = pd.DataFrame(distances_after)
    df2['cell_type'] = df2.index
    df2 = df2.melt("cell_type")

    df = pd.DataFrame()
    df['cell_type'] = df1['cell_type']
    df['channel'] = df1['variable']
    df['EMD_before'] = df1['value']
    df['EMD_after'] = df2['value']

    return df

def plot_emd_scatter(distances_before,distances_after, mode='cell_type'):
    ''''
    Input:
    - distances_before (dict) : dictionary of EMDs before normalization. Computed using 'calculate_emds' function
    - distances_after (dict) : dictionary of EMDs after normalization. Computed using 'calculate_emds' function
    - mode (str) : 'cell_type','celltype_grid','channel' or 'compare' . If 'cell_type', the scatter plot will be colored by cell type.
                    If 'celltype_grid', the scatter plot will be a grid where each cell type is in a subplot
                    If 'channel', the scatter plot will be colored by channel.
                    If 'compare', the scatter will have in red the markers/cell_types that have a higher EMD after normalization than before
                    and in black the markers/cell_types that have a lower EMD after normalization than before

    Returns:
    > a scatter plot of EMDs before and after normalization
    '''
    df = wrap_results(distances_before,distances_after)
    df['bacth correction effect'] = np.where(df['EMD_after'] > df['EMD_before'], 'worsened', 'improved')
    
    if mode == 'compare':
        sns.scatterplot(data=df, y='EMD_before', x='EMD_after',hue='bacth correction effect')
    elif mode == 'channel':
        sns.scatterplot(data=df, y='EMD_before', x='EMD_after',hue='channel')
    elif mode == 'celltype_grid':
        n_celltypes = len(df['cell_type'].unique())
        ncols = 3
        if n_celltypes%ncols == 0:
            nrows = n_celltypes//ncols
        else:
            nrows = n_celltypes//ncols + 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12))
        for i, cell_type in enumerate(df['cell_type'].unique()):
            df_celltype = df.query('cell_type == @cell_type')
            sns.scatterplot(data=df_celltype, y='EMD_before', x='EMD_after',ax=axs[i//3,i%3])
            axs[i//3,i%3].set_title(cell_type)
            axs[i//3,i%3].set_xlabel('EMD after normalization')
            axs[i//3,i%3].set_ylabel('EMD before normalization')
            max_emd = max(df_celltype['EMD_before'].max(),df_celltype['EMD_after'].max())
            x =np.linspace(0, max_emd, 100)
            y = x
            sns.lineplot(x=x, y=y,legend=False, color='#404040', ax=axs[i//3,i%3])
            plt.tight_layout()
        return plt.show()

    else:
        sns.scatterplot(data=df, y='EMD_before', x='EMD_after',hue='cell_type')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Plot a diagonal line
    max_emd = max(df['EMD_before'].max(),df['EMD_after'].max())
    x =np.linspace(0, max_emd, 100)
    y = x
    sns.lineplot(x=x, y=y,color='#404040', legend=False)
    plt.figure(figsize=(5,8))
    return plt.show()

    
