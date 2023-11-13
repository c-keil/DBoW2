#!/bin/bash
# q_iamge=/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul17_5pm_1/cam_3/matlab_clahe2/irpoint_v5/descriptors/1689627191480000019.npy
voc=/media/colin/box_data/ir_data/dbow2_vocabularies/irpoint_k4_L10.yml.gz
q_image=/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul17_5pm_1/cam_3/matlab_clahe2/irpoint_v5/descriptors/1689627188313999891.npy
db_images=/home/colin/Research/ir/DBoW2/example_images/garrage_irpoint_paths_npy.txt
# echo build/simple_ir_test $q_image $db_images $voc test.txt
build/simple_ir_test $q_image $db_images $voc test.txt