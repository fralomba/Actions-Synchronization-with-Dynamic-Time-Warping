import numpy as np
import pandas as pd
import os
import functions
import scipy.io
import matplotlib.pyplot as plt


# LOAD DATA----------------------------------------------------------------------------------
sport_folder = 'PoleVault/'
dataDir = "dataset_dtw/" + sport_folder


finalPaths = pd.read_csv('Paths/PoleVault/path_polevault.csv')
finalPaths = np.array(finalPaths)

# SET ARRAYS---------------------------------------------------------------------------------
names = []
fc7 = []
delete_videos = [34, 35, 39, 54, 75, 87, 96, 97, 112, 113, 136, 137, 141, 145]
hm = []


'''
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
'''

# LOAD PICKLES ANNOTATIONS AND PLOT ----------------------------------------------------------

pickles, g = functions.open_pickles_annots(sport_folder, delete_videos)
print(pickles[1])

milestone_polevault = ['pole_down', 'over_bar', 'end_jump']

real_pole_down = []
stretched_pole_down = []

real_over_bar = []
stretched_over_bar = []

real_end_jump = []
stretched_end_jump = []

print(finalPaths.shape[0])

for i in range(finalPaths.shape[0]):
    real_pole_down.append(pickles[i][milestone_polevault[0]][1])

check = 0
for i in range(finalPaths.shape[0]):
    for k in range(len(finalPaths[i])):
        if check == 0:
            if finalPaths[i][k] == real_pole_down[i]:
                index = k
                check = 1
    check = 0
    stretched_pole_down.append(index)


for i in range(finalPaths.shape[0]):
    real_over_bar.append(pickles[i][milestone_polevault[1]][1])

check = 0
for i in range(finalPaths.shape[0]):
    for k in range(len(finalPaths[i])):
        if check == 0:
            if finalPaths[i][k] == real_over_bar[i]:
                index = k
                check = 1
    check = 0
    stretched_over_bar.append(index)


for i in range(finalPaths.shape[0]):
    real_end_jump.append(pickles[i][milestone_polevault[2]][1])

check = 0
for i in range(finalPaths.shape[0]):
    for k in range(len(finalPaths[i])):
        if check == 0:
            if finalPaths[i][k] == real_end_jump[i]:
                index = k
                check = 1
    check = 0
    stretched_end_jump.append(index)


# NORMALIZE ----------------------------------------------------------------------

stretched_pole_down = [x / len(finalPaths[i]) for x in stretched_pole_down]
stretched_over_bar = [x / len(finalPaths[i]) for x in stretched_over_bar]
stretched_end_jump = [x / len(finalPaths[i]) for x in stretched_end_jump]


real_pole_down_norm = []
for i in range(len(real_pole_down)):
    real_pole_down_norm.append(real_pole_down[i] / max(finalPaths[i]))

real_over_bar_norm = []
for i in range(len(real_pole_down)):
    real_over_bar_norm.append(real_over_bar[i] / max(finalPaths[i]))

real_end_jump_norm = []
for i in range(len(real_pole_down)):
    real_end_jump_norm.append(real_end_jump[i] / max(finalPaths[i]))

# VARIANCE -----------------------------------------------------------------------
print('-----pole down variance:')
print(np.std(real_pole_down_norm))
print(np.std(stretched_pole_down))
print('-----over bar variance:')
print(np.std(real_over_bar_norm))
print(np.std(stretched_over_bar))
print('-----end_jump variance:')
print(np.std(real_end_jump_norm))
print(np.std(stretched_end_jump))

# PLOTS -------------------------------------------------------------------------
X = np.arange(len(real_pole_down))

plt.figure()
plt.title('pole_down')
plt.scatter(X, sorted(real_pole_down_norm))
plt.scatter(X, sorted(stretched_pole_down))
plt.show()

plt.figure()
plt.title('over_bar')
plt.scatter(X, sorted(real_over_bar_norm))
plt.scatter(X, sorted(stretched_over_bar))
plt.show()

plt.figure()
plt.title('end_jump')
plt.scatter(X, sorted(real_end_jump_norm))
plt.scatter(X, sorted(stretched_end_jump))
plt.show()
#
# plt.figure()
# plt.hist(real_pole_down_norm, bins=20)
# plt.title('real_pole_down_norm')
# plt.show()
# plt.figure()
# plt.hist(stretched_pole_down, bins=20)
# plt.title('stretched_pole_down_norm')
# plt.show()
#
# plt.figure()
# plt.hist(real_over_bar_norm, bins=20)
# plt.title('real_over_bar_norm')
# plt.show()
# plt.figure()
# plt.hist(stretched_over_bar, bins=20)
# plt.title('stretched_over_bar_norm')
# plt.show()
#
# plt.figure()
# plt.hist(real_end_jump_norm, bins=20)
# plt.title('real_end_jump_norm')
# plt.show()
# plt.figure()
# plt.hist(stretched_end_jump, bins=20)
# plt.title('stretched_end_jump_norm')
# plt.show()

#functions.global_show_frames(struct, names, samples, finalAnnots, sport_folder)

