import os
import glob
import numpy as np
'''Gathers up and builds list of descriptors for voc generation. Actual voc generation done by calling "build_big_gluestick_voc.sh" 
which calls "DBoW2/build/build_ir_voc"'''

def write_txtfile(name, txt_list):
    mega_str = '\n'.join(txt_list)
    with open(name,'w') as file:
        # _ = [file.write(s + '\n') for s in txt_list]
        file.write(mega_str)


dirs = [
    "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/gluestick_skip2",
    "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick_skip2",
    "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul14_7pm/cam_3/matlab_clahe2/gluestick_skip2",
    "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul14_10pm/cam_3/matlab_clahe2/gluestick_skip2",
    "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul17_5pm_1/cam_3/matlab_clahe2/gluestick_skip2",
    "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day2/cam_3/matlab_clahe2/gluestick_skip2",
    "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/matlab_clahe2/gluestick_skip2",
]

save_path = "/home/colin/Research/ir/DBoW2/vocs/day_night_simple_exp"
os.makedirs(save_path, exist_ok=True)
save_name = "day_night_simple_exp"

all_paths = []
n_descs = []
dirs = [os.path.join(dir, "matched_descriptors") for dir in dirs]
for dir in dirs:
    assert(os.path.isdir(dir))

    paths = sorted(glob.glob(os.path.join(dir, "*.npy")))

    all_paths += paths

    for path in paths:
        descs = np.load(path)
        n_descs.append(len(descs))

n_descs = np.array(n_descs)


s1 = "found {} total desc lists".format(len(all_paths))
s2 = "mean descs per image = {}".format(np.mean(n_descs))
s3 = "total descs = {}".format(np.sum(n_descs))
print(s1)
print(s2)
print(s3)

summary = [s1,s2,s3] + dirs
write_txtfile(os.path.join(save_path,save_name+"_paths.txt"), all_paths)
write_txtfile(os.path.join(save_path,save_name+"_config.txt"), summary)
