import os
import numpy as np
import ndjson 

current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(current_dir, "train")
val_dir = os.path.join(current_dir, "val")
test_dir = os.path.join(current_dir, "test")

file_name = "crowds_students001.ndjson"

def load_data(folder, file_name):
    '''
        {“scene”: 
        {“id”: 266, “p”: 254, “s”: 10238, “e”: 10358, “fps”: 2.5, “tag”: 2}}

        id: scene id
        p: pedestrian ID
        s, e: starting and ending frames id of pedestrian “p”
        fps: frame rate.
        tag: trajectory type. Discussed in detail below.


        {“track”: 
        {“f”: 10238, “p”: 248, “x”: 13.2, “y”: 5.85, “pred_number”: 0, “scene_id”: 123}}

        f: frame id
        p: pedestrian ID
        x, y: x and y coordinates in meters of pedestrian “p” in frame “f”.
        pred_number:    prediction number. This is useful when you are 
                        providing multiple predictions as opposed to a 
                        single prediction. Max 3 predictions allowed
        scene_id:       This is useful when you are providing predictions 
                        of other agents in the scene as opposed to only 
                        primary pedestrian prediction.
    '''

    data_track = []
    data_scene = []
    with open(os.path.join(folder, file_name)) as f:
        reader = ndjson.reader(f)
        for post in reader:
            if 'track' in post.keys():
                data_track.append(post['track'])
            if 'scene' in post.keys():
                data_scene.append(post['scene'])
        return np.asarray(data_track), np.asarray(data_scene)

train_data_track, train_data_scene = load_data(train_dir, file_name)

def convert_track_to_list(data_track):
    '''
        return array with shape: N * max_len * dim
        N is the number of trajectory

        and N * len
    '''
    old_p = 0
    N_traj = []
    N_len = []
    traj = []
    lenth = 0
    for data in data_track:
        if data['p'] == old_p:
            traj.append(np.array([data['x'], data['y']]))  
            lenth += 1
        else:
            old_p = data['p']
            N_traj.append(np.array(traj))
            N_len.append(lenth)
            traj = []
            lenth = 1
            traj.append(np.array([data['x'], data['y']]))
    return N_traj, N_len

tran_traj, train_len =  convert_track_to_list(train_data_track)

def convert_list_to_numpy(traj_list, train_len):
    train_len = np.asarray(train_len)
    max_len = np.max(train_len)
    num_traj = len(traj_list)
    dim_traj = len(traj_list[0][0])
    result = np.zeros([num_traj, max_len, dim_traj])
    for i in range(0, num_traj):
        result[i,:train_len[i], :] = traj_list[i][:,:]

    return result, train_len

result, train_len = convert_list_to_numpy(tran_traj, train_len)

def data_convertion(folder, file_name):
    data_track, data_scene = load_data(folder, file_name)
    data_traj, data_len = convert_track_to_list(data_track)
    return    convert_list_to_numpy(data_traj, data_len)

result, _ = data_convertion(train_dir, file_name)
print(result.shape)

        
        

    

