import os
import glob
import numpy as np
import subprocess
import time

'''Gathers up and builds list of descriptors for voc generation. 

Tests how long it takes to build a voc of various sizes

Actual voc generation done by calling  "DBoW2/build/build_ir_voc"'''

def write_txtfile(name, txt_list):
    mega_str = '\n'.join(txt_list)
    with open(name,'w') as file:
        # _ = [file.write(s + '\n') for s in txt_list]
        file.write(mega_str)

dirs = [
    "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/gluestick_skip2",
    # "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick_skip2",
    # "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul14_7pm/cam_3/matlab_clahe2/gluestick_skip2",
    # "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul14_10pm/cam_3/matlab_clahe2/gluestick_skip2",
    # "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul17_5pm_1/cam_3/matlab_clahe2/gluestick_skip2",
    # "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day2/cam_3/matlab_clahe2/gluestick_skip2",
    # "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/matlab_clahe2/gluestick_skip2",
]

bow_executable = "/home/colin/Research/ir/DBoW2/build/build_ir_voc"
save_path = "/home/colin/Research/ir/DBoW2/vocs/time_experiment2"
os.makedirs(save_path, exist_ok=True)
save_name = "desc_time_exp"
L = 4 # n levels
k = 10 # n branches

experiment_sizes = [20, 200, 2000] #n images

all_paths = []
n_descs = []
dirs = [os.path.join(dir, "matched_descriptors") for dir in dirs]
skip = 200
for dir in dirs:
    assert(os.path.isdir(dir))

    paths = sorted(glob.glob(os.path.join(dir, "*.npy")))


    paths = paths[skip : skip + np.max(experiment_sizes)]
    all_paths += paths


    for path in paths:
        descs = np.load(path)
        n_descs.append(len(descs))

n_descs = np.array(n_descs)


for size in experiment_sizes:
    ds = all_paths[:size]
    nds = n_descs[:size]

    s1 = "found {} total desc lists".format(len(ds))
    s2 = "mean descs per image = {}".format(np.mean(nds))
    s3 = "total descs = {}".format(np.sum(nds))
    print(s1)
    print(s2)
    print(s3)

    save_stub = os.path.join(save_path,save_name + "_{}".format(size))
    desc_dirs = save_stub + "_paths.txt"
    voc_name = save_stub + "_voc_L{}_k{}".format(L,k)
    write_txtfile(desc_dirs , all_paths[:size])

    cmd = [bow_executable, desc_dirs, voc_name, str(k), str(L)]
    start_time = time.time()
    process = subprocess.Popen(cmd)
    process.wait()
    end_time = time.time()
    duration = end_time - start_time
    s4 = "voc creation time = {}".format(duration)
    print(s4)

    summary = [s1,s2,s3,s4] + dirs
    write_txtfile(save_stub + "_summary.txt", summary)
