import os
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import functions
import sys
np.set_printoptions(threshold=sys.maxsize)

from scipy.spatial.distance import euclidean, cosine, sqeuclidean, cityblock, chebyshev
from scipy.cluster.hierarchy import dendrogram, linkage
from dtw import dtw
from numpy import genfromtxt


# LOAD DATA----------------------------------------------------------------------------------
sport_folder = 'PoleVault/'
dataDir = "dataset_dtw/" + sport_folder


sub_structs = np.load('sub_structs/sub_structs_polevault.npy')


childs = pd.read_csv('childs/childs_PoleVault.csv')
childs = np.array(childs)
childs = np.delete(childs, 0, axis=1)

annots_path = "annots/myAnnots.csv"
myAnnots = pd.read_csv(annots_path)
myAnnots = np.array(myAnnots)

# SET ARRAYS---------------------------------------------------------------------------------
names = []
fc7 = []
delete_videos = [34, 35, 39, 54, 75, 87, 96, 97, 112, 113, 136, 137, 141, 145]
milestone_polevault = ['pole_down', 'over_bar', 'end_jump']
hm = []

# COMPUTE FC7, NAMES----------------------------------------------------------------------
i = 0
for file in sorted(os.listdir(dataDir)):
    if file[0] != '.':

        mats = []
        mats = (scipy.io.loadmat(dataDir + file))
        tmp_fc7 = mats['fc7']

        if len(tmp_fc7.shape) > 2:
            names.append(os.path.basename(file.title()).lower())
            if tmp_fc7.shape[0] > 1:
                tmp_fc7 = tmp_fc7[0]

            tmp_fc7 = np.squeeze(tmp_fc7)

            fc7.append(tmp_fc7)

            tmp_hm = np.squeeze(mats['heatmaps'])
            hm.append(tmp_hm)
        else:
            delete_videos.append(i)

        i = i+1

# REARRANGE DATASET  -------------------------------------------------------------------------

finalAnnots = functions.delete_class(myAnnots)
names = functions.string_replace(names)
names = np.delete(names, 122)
fc7 = np.delete(fc7, 122)

n_leafs = len(fc7)
print(n_leafs)

# SELECT SUBTREE
subtree = sub_structs[0:200]

# COMPUTE PATHS AND SHOW FRAMES OF SUBTREE  show = 1 don't showframes
samples, sub_structs = functions.stretch_and_show_subtrees(subtree, n_leafs, childs, finalAnnots, sport_folder, names, show=0)

paths = functions.extract_path_subtrees(sub_structs, samples)

functions.plot_results_polevault(sport_folder, delete_videos, samples, milestone_polevault, paths)

#154
#200 141 168