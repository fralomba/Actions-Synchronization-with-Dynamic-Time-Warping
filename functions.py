import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


from dtw import dtw
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage.transform import resize
from scipy.spatial.distance import euclidean, cosine, sqeuclidean, cityblock, chebyshev
import cv2

from dtw import dtw

import sys
np.set_printoptions(threshold=sys.maxsize)


def get_index_annots(name, finalAnnots):
    index = 0
    i = 0
    while i < finalAnnots.shape[0]:
        if name in finalAnnots[i][0].lower():
            index = i
        i += 1
    return index


def string_replace(names):
    toreplace = '-detectron-feats.mat'

    for i in range(len(names)):
        names[i] = names[i].replace(toreplace, '').replace('v_', '').lower()
    return names


def get_name(i, names):
    name = names[i].lower()
    return name


def delete_class(myAnnots):

    black_list = ['BasketballDunk', 'CricketBowling', 'Fencing', 'FloorGymnastics', 'HorseRiding', 'IceDancing',
                  'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing',
                  'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
    indexes = []

    for k in range(len(black_list)):
        for i in range(myAnnots.shape[0]):
            check = myAnnots[i][0].lower().find(black_list[k].lower())
            if check != -1 :
                indexes.append(i)

    finalAnnots = np.delete(myAnnots, indexes, axis=0)
    return finalAnnots


def show_frames(path, name_x, name_y, start_x, start_y, sport_folder):
    path_img_x = 'video/' + sport_folder + 'v_' + name_x
    path_img_y = 'video/' + sport_folder + 'v_' + name_y

    frames_x = []
    frames_y = []

    for file in sorted(os.listdir(path_img_x)):									#frames video 1
        if file[0] != '.':
            frames_x.append(file)

    for file in sorted(os.listdir(path_img_y)):									#frames video 2
        if file[0] != '.':
            frames_y.append(file)

    for i in range(len(path[0])):												#concatenate the 2 videos e reproduce them
        img_x = cv2.imread(path_img_x + '/' + frames_x[start_x - 1 + path[0][i]])
        img_y = cv2.imread(path_img_y + '/' + frames_y[start_y - 1 + path[1][i]])
        final_img = np.concatenate((img_x, img_y), axis=1)
        final_img = resize(final_img, (500, 700))
        cv2.namedWindow('image')
        cv2.moveWindow('image', 250,180)
        cv2.imshow('image', final_img)
        cv2.waitKey(20)


def global_show_frames(struct, names, samples, finalAnnots, sport_folder):

    if sport_folder == 'PoleVault/':
        if 57 in samples:
            for i in range(len(samples)):
                if samples[i] == 57:
                    index = i

            samples = np.delete(samples, index)
            samples = np.append(samples, [56], axis=0)


    names_samples = []
    indexes = []
    tubes = []
    paths = []

    for i in range(len(samples)):
        names_samples.append(get_name(samples[i], names))
        indexes.append(get_index_annots(names_samples[i], finalAnnots))
        tubes.append([finalAnnots[indexes[i]][1], finalAnnots[indexes[i]][2]])
        paths.append('video/' + sport_folder + 'v_' + names_samples[i])

    frames = [[] for _ in range(len(samples))]

    for i in range(len(samples)):
        for file in sorted(os.listdir(paths[i])):
            if file[0] != '.':
                frames[i].append(file)

    limit = len(struct[samples[0]][1])

    for i in range(limit):
        img = [[] for _ in range(len(samples))]
        for k in range(len(samples)):
            print(k)
            img[k].append(cv2.imread(paths[k] + '/' + frames[k][tubes[k][0] - 1 + struct[samples[k]][1][i]]))

        final_img = np.squeeze(np.concatenate((img), axis=2))
        final_img = resize(final_img, (280, 1000))
        cv2.namedWindow('image')
        cv2.moveWindow('image', 2, 180)
        cv2.imshow('image', final_img)
        cv2.waitKey(1)

        # final_img_0 = np.squeeze(np.concatenate((img[0:4]), axis=2))
        # final_img_0 = resize(final_img_0, (200, 850))
        # final_img_1 = np.squeeze(np.concatenate((img[5:9]), axis=2))
        # final_img_1 = resize(final_img_1, (200, 850))
        
        # final_img = np.concatenate((final_img_0, final_img_1), axis=0)
        # cv2.namedWindow('image')
        # cv2.moveWindow('image', 10, 10)
        # cv2.imshow('image', final_img)
        # cv2.waitKey(20)


def stretch_path(node, struct, childs):

        # Figlio sx
        path_sx = childs[node][0]

        if struct[path_sx][1] == 0:
            struct[path_sx][1] = struct[node][1][0]
        else:

            tmp_path_sx = struct[path_sx][1][0]
            tmp_path_dx = struct[path_sx][1][1]

            stretch_path_sx = tmp_path_sx[struct[node][1][0]]
            stretch_path_dx = tmp_path_dx[struct[node][1][0]]

            struct[path_sx][1] = [stretch_path_sx, stretch_path_dx]

        # Figlio destro
        path_dx = childs[node][1]

        if struct[path_dx][1] == 0:
            struct[path_dx][1] = struct[node][1][1]
        else:

            tmp_path_sx = struct[path_dx][1][0]
            tmp_path_dx = struct[path_dx][1][1]

            stretch_path_sx = tmp_path_sx[struct[node][1][1]]
            stretch_path_dx = tmp_path_dx[struct[node][1][1]]

            struct[path_dx][1] = [stretch_path_sx, stretch_path_dx]

        return struct


def symmetrize(matrix):
    return matrix + matrix.T - np.diag(matrix.diagonal())


def open_pickles_annots(sport_folder, delete_videos):

    pickles_folder_path = 'new_Annots/' + sport_folder + '/'
    pickles = []
    bad_indexes = []
    subdirs = [x[0] for x in sorted(os.walk(pickles_folder_path))]

    i = 0
    for subdir in subdirs:
        for file in sorted(os.listdir(subdir)):
            if file[0] == '0':
                if i not in delete_videos:
                    check = pickle.load(open(subdir + '/' + file, 'rb'))
                    if len(check) == 3:
                        pickles.append(pickle.load(open(subdir + '/' + file, 'rb')))
                    else:
                        bad_indexes.append(i)
                i = i + 1

    bad_indexes = [x - len(delete_videos) for x in bad_indexes]

    return pickles, bad_indexes

def create_distance_matrix(fc7, sport_folder):

    videos = np.arange(len(fc7))

    print('number of videos:', len(fc7))

    n = len(videos)

    distMatrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if j <= i :
                dist, cost, acc, path = dtw(fc7[videos[i]], fc7[videos[j]], dist=sqeuclidean)
                distMatrix[i][j] = dist

        print('pass', i,'of', n-1)

    print('writing distance matrix...')
    distMatrix = symmetrize(distMatrix)

    dm = pd.DataFrame(distMatrix)
    dm.to_csv('distMatrix/' + sport_folder[0 : -1] + '_dist.csv', sep=',', encoding='utf-8')

    print('done.')


def make_dendrogram(distMatrix, plot=0):
    Z = linkage(distMatrix, 'ward')  # hierarcical clustering
    print(Z)
    if plot == 1:
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')  # plot relative dendrogram
        plt.ylabel('distance')
        dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
        plt.show()

    return Z


def compute_paths(Z, struct, sub_structs, childs):
    best = []
    zero = [0]

    for i in range(Z.shape[0]):

        stretch_fc7 = []

        index_a = Z[i, 0].astype(int)
        index_b = Z[i, 1].astype(int)

        a = struct[index_a][0]
        b = struct[index_b][0]

        childs = np.append(childs, [[index_a, index_b]], axis=0)

        dist, cost, acc, path = dtw(a, b, dist=sqeuclidean)

        # Metrica = 1: più lungo, 2: media
        metrica = 1

        if metrica == 1:
            print('Length >>')
            if a.shape[0] > b.shape[0]:
                best = a
                id = 0
            else:
                best = b
                id = 1

            stretch_fc7 = best[path[id], :]

        if metrica == 2:
            print('Media')
            to_metrics_1 = a[path[0], :]
            to_metrics_2 = b[path[1], :]
            stretch_fc7 = (to_metrics_1 + to_metrics_2) / 2

        struct = np.append(struct, [[stretch_fc7, path]], axis=0)
        sub_structs = np.append(sub_structs, [[zero, path]], axis=0)
        print(path)

    return struct, sub_structs, childs


def stretch_and_show_subtrees(sub_structs, n_leafs, childs, finalAnnots, sport_folder, names, show = 0):

    samples = []

    for i in range(sub_structs.shape[0] - 1, n_leafs - 1, -1):
        sub_structs = stretch_path(i, sub_structs, childs)

    samples = find_leafs(childs, len(sub_structs)-1, n_leafs)
    samples = flatten(samples)

    if show == 0:
        global_show_frames(sub_structs, names, samples, finalAnnots, sport_folder)

    print(samples)

    return samples, sub_structs



def find_leafs(childs, node, n_leafs):

    leafs =[]

    if childs[node][0] < n_leafs:
        leafs.append(childs[node][0])
    else:
        leafs.append(find_leafs(childs, childs[node][0], n_leafs))

    if childs[node][1] < n_leafs:
        leafs.append(childs[node][1])
    else:
        leafs.append(find_leafs(childs, childs[node][1], n_leafs))

    return leafs


def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis


def plot_results_polevault(sport_folder, delete_videos, samples, milestone_polevault, paths):

    all_pickles, f = open_pickles_annots(sport_folder, delete_videos)

    pickles= []
    for x in samples:
        pickles.append(all_pickles[x])

    #++++++++++++++++++++++++++++++++++++++++++++++++
    real_pole_down = []
    stretched_pole_down = []

    real_over_bar = []
    stretched_over_bar = []

    real_end_jump = []
    stretched_end_jump = []

    for i in range(len(paths)):
        real_pole_down.append(pickles[i][milestone_polevault[0]][1])

    check = 0
    for i in range(len(paths)):
        for k in range(len(paths[i])):
            if check == 0:
                if paths[i][k] == real_pole_down[i]:
                    index = k
                    check = 1
        check = 0
        stretched_pole_down.append(index)

    for i in range(len(paths)):
        real_over_bar.append(pickles[i][milestone_polevault[1]][1])

    check = 0
    for i in range(len(paths)):
        for k in range(len(paths[i])):
            if check == 0:
                if paths[i][k] == real_over_bar[i]:
                    index = k
                    check = 1
        check = 0
        stretched_over_bar.append(index)

    for i in range(len(paths)):
        real_end_jump.append(pickles[i][milestone_polevault[2]][1])

    check = 0
    for i in range(len(paths)):
        for k in range(len(paths[i])):
            if check == 0:
                if paths[i][k] == real_end_jump[i]:
                    index = k
                    check = 1
        check = 0
        stretched_end_jump.append(index)

    # NORMALIZE ----------------------------------------------------------------------

    stretched_pole_down = [x / len(paths[i]) for x in stretched_pole_down]
    stretched_over_bar = [x / len(paths[i]) for x in stretched_over_bar]
    stretched_end_jump = [x / len(paths[i]) for x in stretched_end_jump]


    real_pole_down_norm = []
    for i in range(len(real_pole_down)):
        real_pole_down_norm.append(real_pole_down[i] / max(paths[i]))

    real_over_bar_norm = []
    for i in range(len(real_pole_down)):
        real_over_bar_norm.append(real_over_bar[i] / max(paths[i]))

    real_end_jump_norm = []
    for i in range(len(real_pole_down)):
        real_end_jump_norm.append(real_end_jump[i] / max(paths[i]))

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

    # plt.figure()
    # plt.hist(real_pole_down_norm)
    # plt.title('real_pole_down_norm')
    # plt.show()
    # plt.figure()
    # plt.hist(stretched_pole_down)
    # plt.title('stretched_pole_down_norm')
    # plt.show()

    # plt.figure()
    # plt.hist(real_over_bar_norm)
    # plt.title('real_over_bar_norm')
    # plt.show()
    # plt.figure()
    # plt.hist(stretched_over_bar)
    # plt.title('stretched_over_bar_norm')
    # plt.show()

    # plt.figure()
    # plt.hist(real_end_jump_norm)
    # plt.title('real_end_jump_norm')
    # plt.show()
    # plt.figure()
    # plt.hist(stretched_end_jump)
    # plt.title('stretched_end_jump_norm')
    # plt.show()


def extract_path_subtrees(sub_structs, samples):

    paths = []

    for x in samples:
        paths.append(sub_structs[x][1])

    return paths


def plot_results_diving(sport_folder, delete_videos, samples, milestone_diving, paths):

    all_pickles, f = open_pickles_annots(sport_folder, delete_videos)

    pickles = []
    for x in samples:
        pickles.append(all_pickles[x])

    real_skip = []
    stretched_skip = []

    real_top_flight = []
    stretched_top_flight = []

    real_entrance_in_water = []
    stretched_entrance_in_water = []

    for i in range(len(paths)):
        real_skip.append(pickles[i][milestone_diving[0]][1])

    check = 0
    for i in range(len(paths)):
        for k in range(len(paths[i])):
            if check == 0:
                if paths[i][k] == real_skip[i]:
                    index = k
                    check = 1
        check = 0
        stretched_skip.append(index)

    for i in range(len(paths)):
        real_top_flight.append(pickles[i][milestone_diving[1]][1])

    check = 0
    for i in range(len(paths)):
        for k in range(len(paths[i])):
            if check == 0:
                if paths[i][k] == real_top_flight[i]:
                    index = k
                    check = 1
        check = 0
        stretched_top_flight.append(index)

    for i in range(len(paths)):
        real_entrance_in_water.append(pickles[i][milestone_diving[2]][1])

    check = 0
    for i in range(len(paths)):
        for k in range(len(paths[i])):
            if check == 0:
                if paths[i][k] == real_entrance_in_water[i]:
                    index = k
                    check = 1
        check = 0
        stretched_entrance_in_water.append(index)

    # NORMALIZE ----------------------------------------------------------------------

    stretched_skip = [x / len(paths[i]) for x in stretched_skip]
    stretched_top_flight = [x / len(paths[i]) for x in stretched_top_flight]
    stretched_entrance_in_water = [x / len(paths[i]) for x in stretched_entrance_in_water]


    real_skip_norm = []
    for i in range(len(real_skip)):
        real_skip_norm.append(real_skip[i] / max(paths[i]))

    real_top_flight_norm = []
    for i in range(len(real_top_flight)):
        real_top_flight_norm.append(real_top_flight[i] / max(paths[i]))

    real_entrance_in_water_norm = []
    for i in range(len(real_entrance_in_water)):
        real_entrance_in_water_norm.append(real_entrance_in_water[i] / max(paths[i]))

    # VARIANCE -----------------------------------------------------------------------
    print('-----pole down variance:')
    print(np.std(real_skip_norm))
    print(np.std(stretched_skip))
    print('-----over bar variance:')
    print(np.std(real_top_flight_norm))
    print(np.std(stretched_top_flight))
    print('-----end_jump variance:')
    print(np.std(real_entrance_in_water_norm))
    print(np.std(stretched_entrance_in_water))

    # PLOTS -------------------------------------------------------------------------

    X = np.arange(len(real_skip_norm))


    plt.figure()
    plt.title('Skip')
    plt.scatter(X, sorted(real_skip_norm))
    plt.scatter(X, sorted(stretched_skip))
    plt.show()

    plt.figure()
    plt.title('Top_flight')
    plt.scatter(X, sorted(real_top_flight_norm))
    plt.scatter(X, sorted(stretched_top_flight))
    plt.show()

    plt.figure()
    plt.title('entrance_in_water')
    plt.scatter(X, sorted(real_entrance_in_water_norm))
    plt.scatter(X, sorted(stretched_entrance_in_water))
    plt.show()

    # plt.figure()
    # plt.hist(real_skip_norm)
    # plt.title('real_skip_norm')
    # plt.show()
    # plt.figure()
    # plt.hist(stretched_skip)
    # plt.title('stretched_skip_norm')
    # plt.show()

    # plt.figure()
    # plt.hist(real_top_flight_norm)
    # plt.title('real_top_flight_norm')
    # plt.show()
    # plt.figure()
    # plt.hist(stretched_top_flight)
    # plt.title('stretched_top_flight_norm')
    # plt.show()

    # plt.figure()
    # plt.hist(real_entrance_in_water_norm)
    # plt.title('real_entrance_in_water_norm')
    # plt.show()
    # plt.figure()
    # plt.hist(stretched_entrance_in_water)
    # plt.title('stretched_entrance_in_water_norm')
    # plt.show()






