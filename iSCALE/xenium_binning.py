import pickle
from PIL import Image
from einops import reduce
from einops import rearrange
import numpy as np
import pandas as pd
import sys
import itertools

spot_size=16
scale = 0.5669290981461985

hist = pickle.load(open('embeddings-hist.pickle', 'rb'))
cls1 = np.array(hist['cls'])
sub1 = np.array(hist['sub'])
hist1 = np.concatenate((cls1, sub1), 0)
#hist1 = np.flip(np.flip(hist1,1),2)


Image.MAX_IMAGE_PIXELS = None
transcripts = pd.read_csv('transcripts_registered.csv', index_col = 0)
transcripts.x_location = transcripts.x_location*scale
transcripts.y_location = transcripts.y_location*scale

hist1_2 = hist1[:,int(np.floor(transcripts.y_location.min()/16)):int((np.ceil(transcripts.y_location.max()/16)//16+1) * 16),int(np.floor(transcripts.x_location.min()/16)):int((np.ceil(transcripts.x_location.max()/16)//16 +1) * 16)]
hist1 = np.stack([reduce(i, '(h1 h) (w1 w) -> h1 w1', 'mean', h=int(spot_size/16), w=int(spot_size/16)) for i in hist1_2])
hist2 = rearrange(hist1, 'c h w -> h w c')
hist3 = rearrange(hist2, 'h w c-> (h w) c')

transcripts.x_location = transcripts.x_location-np.floor(transcripts.x_location.min()/16)*16
transcripts.y_location = transcripts.y_location-np.floor(transcripts.y_location.min()/16)*16                               

transcripts['bin_x'] = np.floor(transcripts.x_location/spot_size).astype('int')
transcripts['bin_y'] = np.floor(transcripts.y_location/spot_size).astype('int')
transcripts1 = transcripts[['feature_name', 'bin_x', 'bin_y']]
transcripts1['count'] = 1
spots = transcripts1.groupby(['bin_x', 'bin_y','feature_name']).agg('sum').reset_index()
spots = spots[spots.bin_x>=0]
cols = ['bin_x', 'bin_y']
spots['id'] = spots[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
spots = spots[[i.split('_')[0] not in ['BLANK','NegControlCodeword','NegControlProbe'] for i in spots.feature_name]]
spots1 = spots.pivot(index='id', columns = 'feature_name', values = 'count')
spots1[np.isnan(spots1)] = 0
spots2 = pd.DataFrame(0, index = [str(i[0]) + '_' + str(i[1]) for i in list(itertools.product(range(hist1.shape[2]), range(hist1.shape[1])))], columns = spots1.columns)
spots2.loc[spots1.index] = spots1.values
spots2.index = [i.split('_')[1] + '_' + i.split('_')[0] for i in spots2.index]




locs = [(i,j) for i in range(hist2.shape[0]) for j in range(hist2.shape[1])]
x_locs = [i[0] for i in locs]
y_locs = [i[1] for i in locs]
x_locs1 = [int((i+1)*spot_size-spot_size/2) for i in x_locs]
y_locs1 = [int((i+1)*spot_size-spot_size/2) for i in y_locs]
xy_locs = [(i,j) for i,j in zip(x_locs1,y_locs1)]
locs = pd.DataFrame(index=range(len(xy_locs)))
locs['x'] = x_locs
locs['y'] = y_locs
locs['x_pixel'] = [i[0] for i in xy_locs]
locs['y_pixel'] = [i[1] for i in xy_locs]
locs.index = [str(x_locs[i]) + '_' + str(y_locs[i]) for i in range(len(x_locs))]


#mask = Image.open('mask-small.png')
#mask = mask.resize((mask.size[0]*16, mask.size[1]*16))
#mask1 = np.array(mask)
#mask_bool = [mask1[i] for i in xy_locs]

spots3 = spots2.loc[locs.index]
#locs = locs[mask_bool]
#spots3 = spots3[mask_bool]
ge = spots3.values
#hist3 = hist3[mask_bool]

ge = ge[~np.isnan(hist3.sum(1))]
locs = locs[~np.isnan(hist3.sum(1))]
hist3 = hist3[~np.isnan(hist3.sum(1))]


np.save('rna_' + str(spot_size) + '.npy', ge)
np.save('hist_' + str(spot_size) + '.npy', hist3)
locs.to_csv('locs_' + str(spot_size) + '.csv')

