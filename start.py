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


# LOAD DATASET -------------------------------------------------------------------------
sport_folder = 'Diving/'
dataDir = "dataset_dtw/" + sport_folder

# DECLARE ARRAYS ------------------------------------------------------------------------
fc7 = []
hm = []
names = []
delete_videos = []
struct = []
sub_structs = []

annots_path = "annots/myAnnots.csv"
myAnnots = pd.read_csv(annots_path)
myAnnots = np.array(myAnnots)


# LOAD FEATURES FC7 -----------------------------------------------------------------------

print('start loading data...')

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

print(delete_videos)

# REARRANGE DATASET  -------------------------------------------------------------------------

finalAnnots = functions.delete_class(myAnnots)
names = functions.string_replace(names)
pickles, bad_indexes = functions.open_pickles_annots(sport_folder, delete_videos)
names = np.delete(names, bad_indexes)
fc7 = np.delete(fc7, bad_indexes)
print(bad_indexes)

'''

for i in range(n):                                                                              #
    for j in range(n):                                                                          #
        if j <= i:                                                                              #
            dist, cost, acc, path = dtw(fc7[samples[i]], fc7[samples[j]], dist=sqeuclidean)     # compute a local
            distMatrix[i][j] = dist                                                             # distance matrix
    struct.append([fc7[samples[i]], 0])                                                         #
    print('pass', i, 'of', n-1)                                                                 #

print('writing distance matrix...')
distMatrix = functions.symmetrize(distMatrix)
'''

# INITIALIZE TREE STRUCT ---------------------------------------------------------------

samples = np.arange(len(fc7))     # index of videos

for i in range(len(samples)):
    struct.append([fc7[i], 0])
    sub_structs.append([0, 0])

childs = []

# Initialize childs with leafs
for i in range(len(samples)):
    childs.append([0, 0])
childs = np.array(childs)

# IMPORT MATRIX DISTANCES - CLUSTERING AND PLOT DENDROGRAM-------------------------------

distMatrix = pd.read_csv('distMatrix/' + sport_folder[0:-1] + '_dist.csv')
distMatrix = np.array(distMatrix)

Z = functions.make_dendrogram(distMatrix, 1)   # hierarcical clustering and plot


# COMPUTE PATHS --------------------------------------------------------------------------

struct = np.array(struct)
sub_structs = np.array(sub_structs)

struct, sub_structs, childs = functions.compute_paths(Z, struct, sub_structs, childs)

#--------------------------Save father paths (sub-structs)---------------------------------------
#
np.save('sub_structs/sub_structs_diving.npy', sub_structs)
print('end save substruct')

for i in range(struct.shape[0]-1, len(samples)-1, -1):
    struct = functions.stretch_path(i, struct, childs)

#-----------------------------Save finals Path--------------------------------------

# tosave = []
# for i in range(len(samples)):
#     tosave.append(struct[i][1])
#
# tosave = pd.DataFrame(tosave)
# tosave.to_csv('Paths/PoleVault/path_palevault.csv')

print('writing childs...')
childs = pd.DataFrame(childs)
childs.to_csv('childs/childs_diving' + sport_folder[0:-1] + '.csv', sep=',', encoding='utf-8')
print('done.')

#-------------------------------------PICKLES--------------------------------------
#
# milestone_polevault = ['pole_down', 'over_bar', 'end_jump']
#
# real_pole_down = []
# stretched_pole_down = []
#
# real_over_bar = []
# stretched_over_bar = []
#
# real_end_jump = []
# stretched_end_jump = []
#
#
# for i in range(len(samples)):
#     real_pole_down.append(pickles[samples[i]][milestone_polevault[0]][1])
#
#
# check = 0
# for i in range(len(samples)):
#     for k in range(len(struct[i][1])):
#         if check == 0:
#             if struct[i][1][k] == real_pole_down[i]:
#                 index = k
#                 check = 1
#     check = 0
#     stretched_pole_down.append(index)
#
#
# for i in range(len(samples)):
#     real_over_bar.append(pickles[samples[i]][milestone_polevault[1]][1])
#
# check = 0
# for i in range(len(samples)):
#     for k in range(len(struct[i][1])):
#         if check == 0:
#             if struct[i][1][k] == real_over_bar[i]:
#                 index = k
#                 check = 1
#     check = 0
#     stretched_over_bar.append(index)
#
#
# for i in range(len(samples)):
#     real_end_jump.append(pickles[samples[i]][milestone_polevault[2]][1])
#
# check = 0
# for i in range(len(samples)):
#     for k in range(len(struct[i][1])):
#         if check == 0:
#             if struct[i][1][k] == real_end_jump[i]:
#                 index = k
#                 check = 1
#     check = 0
#     stretched_end_jump.append(index)
#
#
# samples = [29, 88]
# functions.global_show_frames(struct, names, samples, finalAnnots, sport_folder)
#
# a = np.arange(len(stretched_pole_down))
#
# print(np.var(real_pole_down))
# print(np.var(stretched_pole_down))
#
# stretched_pole_down = [x / len(struct[0][1]) for x in stretched_pole_down]
# stretched_over_bar = [x / len(struct[0][1]) for x in stretched_over_bar]
# stretched_end_jump = [x / len(struct[0][1]) for x in stretched_end_jump]
#
#
# real_pole_down_norm = []
# for i in range(len(real_pole_down)):
#     real_pole_down_norm.append(real_pole_down[i] / max(struct[i][1]))
#
# real_over_bar_norm = []
# for i in range(len(real_pole_down)):
#     real_over_bar_norm.append(real_over_bar[i] / max(struct[i][1]))
#
# real_end_jump_norm = []
# for i in range(len(real_pole_down)):
#     real_end_jump_norm.append(real_end_jump[i] / max(struct[i][1]))
#
#
# plt.figure()
# plt.title('Results')
# plt.plot(real_pole_down_norm)
# plt.plot(stretched_pole_down)
# plt.show()
#
# plt.figure()
# plt.title('Results')
# plt.plot(real_over_bar_norm)
# plt.plot(stretched_over_bar)
# plt.show()
#
# plt.figure()
# plt.title('Results')
# plt.plot(real_end_jump_norm)
# plt.plot(stretched_end_jump)
# plt.show()
#
# plt.figure()
# plt.hist(real_pole_down_norm, bins=25)
# plt.show()
# plt.figure()
# plt.hist(stretched_pole_down, bins=25)
# plt.show()
#
# plt.figure()
# plt.hist(real_over_bar_norm, bins=25)
# plt.show()
# plt.figure()
# plt.hist(stretched_over_bar, bins=25)
# plt.show()
#
# plt.figure()
# plt.hist(real_end_jump_norm, bins=25)
# plt.show()
# plt.figure()
# plt.hist(stretched_end_jump, bins=25)
# plt.show()
#
#
# Y = np.arange(len(real_end_jump_norm))/10
# plt.figure()
# plt.scatter(real_end_jump_norm, Y)
# plt.show()
# plt.figure()
# plt.scatter(stretched_end_jump, Y)
# plt.show()
