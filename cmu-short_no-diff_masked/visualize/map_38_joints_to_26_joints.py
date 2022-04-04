import numpy as np

# from processor/data_tools.py
dim_to_ignore = [0,  1,  2,  3,  4,  5,  6,   7,   8,   21,  22,  23,  24,  25,  26, 
    39, 40, 41, 60, 61, 62, 63,  64,  65,  81,  82,  83,
    87, 88, 89, 90, 91, 92, 108, 109, 110, 114, 115, 116]
dim_to_use = [9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31,  32,  33,  34,  35,  36,  37,  38, 
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,  59,  66,  67,  68,  69,  70,  71,  72,  73,  74, 
    75, 76, 77, 78, 79, 80, 84, 85, 86, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 111, 112, 113]

# get indices of joints ignored
# (within the 39-joint skeletal model)
joint_ids_ignored = []
for i in range(0, len(dim_to_ignore), 3):
    joints_38_index = int(dim_to_ignore[i]/3 - 1)
    if joints_38_index != -1:
        joint_ids_ignored.append(joints_38_index)
# print(joint_ids_ignored)

# build mapping from 38-joint-skeleton index to 26-joint
# skeleton index and vice versa
joints_38 = np.zeros([38, ], dtype=np.int32)
joints_26 = np.zeros([26, ], dtype=np.int32)
joints_26_index = 0
for joints_38_index in range(38):
    if joints_38_index in joint_ids_ignored:
        joints_38[joints_38_index] = -1
    else:
        joints_38[joints_38_index] = joints_26_index
        joints_26[joints_26_index] = joints_38_index
        joints_26_index += 1

# see for each joint in the 26-joint model, which joint it
# corresponds to in the 38-joint model
# for i in range(26):
#     print(joints_26[i])
# the other way round
# for i in range(38):
#     print(joints_38[i])