import os
import torch
# logStr = "def\n"
# # with open("train_log.txt", "w") as f:
# #     f.write(logStr)
# f = open("train_log.txt", "a")
# f.write(logStr)
# f.close()

# if (os.path.exists("train_log.txt")):
#     with open("train_log.txt", 'rb') as f:
#         try:  # catch OSError in case of a one line file
#             f.seek(-2, os.SEEK_END)
#             while f.read(1) != b'\n':
#                 f.seek(-2, os.SEEK_CUR)
#         except OSError:
#             f.seek(0)
#         last_line = f.readline().decode()
#         num = int(last_line.split("_")[0].split("=")[1].strip())
#         max_acc = float(last_line.split("_")[4].split("=")[1].strip())
#         min_loss = float(last_line.split("_")[3].split("=")[1].strip())
#         print(last_line)
#         print(num)
#         print(max_acc)
#         print(min_loss)


w = torch.randn(2,3,4)
width = w.size(2)
height = w.size(1)
w2 = w[:, 0:height-1, 0:width-1]
w3 = w[:, 1:height, 1:width]
print(w)
print(w2)
print(w3)
print(width, height)

@staticmethod
def _point_on_track(distance, track, offset=0.0):
    """
    Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
    Returns a 3d coordinate
    """
    
    # print(distance)
    
    path_distances=track.path_distance
    
    if track.path_distance.shape[0]==386:
        num_nodes=222
    else:
        num_nodes=len(path_distances)
    max_distance=path_distances[num_nodes-1][1]

    # print(distance)
    ####### 1 Get node_idx 
    node_idx = np.searchsorted(path_distances[:num_nodes, 1], distance%max_distance) %num_nodes
    
    
    ####### 2 Get node_idx 
    # min_difference=100000000
    # node_idx_for_min=None
    # for node_idx in range(num_nodes):
    #     if abs(track.path_distance[node_idx,1]-distance)<min_difference:
    #         node_idx_for_min=node_idx
    #         min_difference=abs(track.path_distance[node_idx,1]-distance)
        
    # node_idx=node_idx_for_min
    
    ####### 3 Get node_idx 
    # sorted_distances=np.sort(track.path_distance[...,1])
    # node_idx_for_sorted = np.searchsorted(sorted_distances, distance%max_distance) %num_nodes
    # target_distance=sorted_distances[node_idx_for_sorted]
    # node_idx=np.where(track.path_distance[...,1]==target_distance)[0].item()

    # print(node_idx)
    d = track.path_distance[node_idx]
    # print(node_idx,d)
    x = track.path_nodes[node_idx]
    
    global distance_g2
    distance_g2=distance
    # print(distance)
    
    global track_g
    track_g=track
    
    t = (distance + offset - d[0]) / (d[1] - d[0])