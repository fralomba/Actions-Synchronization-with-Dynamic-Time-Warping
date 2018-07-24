from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pickle


# helper function wrapping cv2.putText with default values
def puttext(im, title, coords, size=0.4, color=(255, 255, 255)):
    cv2.putText(im, title, coords, cv2.FONT_HERSHEY_SIMPLEX, size, color)
    return


def save_tube_annot(tube_annot):
    if not os.path.isdir(out_folder + '/' + video_name):
        os.makedirs(out_folder + '/' + video_name)
    tube_save_path = out_folder + '/' + video_name + '/' + str(tube_id) + '.pickle'
    # np.save(tube_save_path, tube_annot)
    with open(tube_save_path, 'wb') as handle:
        pickle.dump(tube_annot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


cls = 'Diving'
annot_types = {'1': 'skip', '2': 'top_flight', '3': 'entrance_in_water'}

dataset_path = 'video/'
annot_path = 'annots/'
out_folder = 'new_annots/'

annots = sio.loadmat(annot_path + 'finalAnnots', struct_as_record=True)['annot'][0] # all annotations
cur_annots = [x for x in annots if cls == x[1][0].split('/')[0]] # take only annotations of current class

draw_bb = False
for a in cur_annots:
    num_frames = a[0]
    video_name = a[1][0]
    tubes = a[2][0]
    num_tubes = len(tubes)
    for tube_id in range(num_tubes):
        start_frame = tubes['sf'][tube_id][0][0]
        end_frame = tubes['ef'][tube_id][0][0]
        boxes = tubes['boxes'][tube_id].astype(np.int32)
        assert(boxes.shape[0] == end_frame - start_frame + 1)

        frame_id = start_frame
        done = False

        if os.path.isfile(out_folder + '/' + video_name + '/' + str(tube_id) + '.pickle'):
            # tube_annot = np.load(out_folder + '/' + video_name + '/' + str(tube_id) + '.npy')
            with open(out_folder + '/' + video_name + '/' + str(tube_id) + '.pickle', 'rb') as handle:
                tube_annot = pickle.load(handle)
        else:
            tube_annot = {}

        # frame names are numerated matlab style
        while not done:
            # frame index
            index_in_tube = frame_id - start_frame
            frame_key = video_name + '/' + str(tube_id) + '_' + str(index_in_tube) + '_' + str(frame_id)

            cur_action = None
            annot_items = tube_annot.items() # key, val
            keys_ = [a[0] for a in annot_items]
            vals_ = [a[1] for a in annot_items]
            if [frame_id, index_in_tube] in vals_:
                cur_action = (keys_[np.where([x == [frame_id, index_in_tube] for x in vals_])[0][0]])

            # read image
            im = cv2.imread(dataset_path + video_name + '/' + str(frame_id).zfill(5) + '.jpg')

            # show bounding box
            if draw_bb:
                box = boxes[index_in_tube]
                cv2.rectangle(im, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 3)

            # shitty way to add a title to an image with opencv
            im = np.pad(im, ((40,0), (0,0), (0,0)), 'constant', constant_values=0)
            puttext(im, video_name + ' - tube: ' + str(tube_id), (int(im.shape[0]/2) - 100, 15))
            puttext(im, 'FRAME ' + str(index_in_tube), (int(im.shape[0]/2) - 100, 35))
            if cur_action is not None:
                puttext(im, cur_action, (int(im.shape[0]/2), 35))
            cv2.imshow('frame', im)

            key = cv2.waitKey(0)
            # next frame
            if key == ord('d'):
                frame_id = min(frame_id + 1, end_frame)
            # prev frame
            elif key == ord('a'):
                frame_id = max(frame_id - 1, start_frame)
            # draw boxes
            elif key == ord('b'):
                draw_bb = not draw_bb
            # next tube
            elif key == 13:
                save_tube_annot(tube_annot)
                done = True
            # numbers 0-9
            elif 48 <= key <= 57:
                key = str(key - 48)
                if key in annot_types.keys():
                    tube_annot[annot_types[key]] = [frame_id, index_in_tube]
            # ESC
            elif key == 27:
                print('bye')
                raise SystemExit(0)
