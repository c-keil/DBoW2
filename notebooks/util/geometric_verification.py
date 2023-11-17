import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

fx = 524.888150
fy = 521.776791
cx = 325.5969897
cy = 242.392342

k1 = -0.470302508
k2 = 0.3010578604
p1 = 0.00468835914
p2 = -0.00165573977
p3 = -0.12437411

K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
dist_coefs =  np.array([k1, k2, p1, p2])
# dist_coefs =  np.array([k1, k2, p1, p2, p3])

def extract_timestamp(string):
    '''reads the string timestamps out of a file path'''
    name = os.path.basename(string)
    name = '.'.join(name.split('.')[:-1])
    proposed_stamp = name.split('_')[0]
    # print(string)
    # print(proposed_stamp)
    # stamp_length = 19
    # proposed_stamp = name[:stamp_length]
    assert(''.join(proposed_stamp.split('.')).isnumeric())
    return proposed_stamp

def is_image_file(file):
    assert(os.path.isfile(file))
    return (file[-4:] == ".png") or (file[-4:] == ".jpg")

def pixel_dist(kp1, kp2):
    return np.sqrt((kp1.pt[0]-kp2.pt[0])**2 + (kp1.pt[1]-kp2.pt[1])**2)

def label_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (5, 30)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    return cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

# def readKp(strings):
#     assert(len(strings) == 7)
#     kp = cv2.KeyPoint()
#     kp.pt = (float(strings[0]), float(strings[1]))
#     kp.size = float(strings[2])
#     kp.angle = float(strings[3])
#     kp.response = float(strings[4])
#     kp.octave = int(float(strings[5]))
#     kp.class_id = int(float(strings[6]))
#     return kp

# def readDesc(strings):
#     assert(len(strings) == 256)
#     nums = [float(i) for i in strings]
#     return nums

def kp_array_to_cv2(kp_array):
    return [cv2.KeyPoint(k[0], k[1], 1.0) for k in kp_array]

def readKpFile(kp_file):
    data = np.load(kp_file)
    assert(data.dtype == np.float32)
    kpts = kp_array_to_cv2(data)
    return kpts

def readDescFile(desc_file):
    descriptors = np.load(desc_file)
    assert(descriptors.dtype == np.float32)
    return descriptors 

def undistort_points(pts):
    assert(type(pts) == np.ndarray)
    assert(len(pts.shape) == 2)
    return cv2.undistortPoints(pts.reshape(-1,1,2), K, dist_coefs, np.array([]),K).reshape(-1,2)

def undistort_keypoints(kps):
    pts_npy = np.array([[[k.pt[0], k.pt[1]]] for k in kps])
    pts_un = cv2.undistortPoints(pts_npy, K, dist_coefs, np.array([]),K)
    kps_un = [cv2.KeyPoint(p[0],p[1],1.0) for p in pts_un.reshape(-1,2)]
    return kps_un

def match_descriptors(kp1, desc1, kp2, desc2, orb = False):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if orb:
        bf = cv2.BFMatcher(crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # print(dir(bf))
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches

def match_descriptors_l(kp1, desc1, kp2, desc2, l = None):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    # use lowes ratio
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(desc1, desc2)

    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches

def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers

def compute_fundamental(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    F, inliers = cv2.findFundamentalMat(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
                                    
    inliers = inliers.flatten()
    return F, inliers

def get_match_distances(desc1, desc2, matches):
    return np.array([np.linalg.norm(desc1[match.queryIdx] - desc2[match.trainIdx]) for match in matches])

def get_pixel_distances(kp1, kp2, matches):
    return np.array([pixel_dist(kp1[match.queryIdx],kp2[match.trainIdx]) for match in matches])

def get_files(file, desc_dir="matched_descriptors", kp_dir="matched_keypoints"):
    '''At the moment, this assumes a descriptor file input'''
    assert os.path.isfile(file)

    basename = os.path.basename(file)
    parent_dir = os.path.dirname(os.path.dirname(file))
    image_dir = os.path.dirname(parent_dir)
    desc_file = file
    kp_file = os.path.join(parent_dir, kp_dir, basename)
    im_file = os.path.join(image_dir, basename[:-4]+".png")
    
    assert os.path.isfile(kp_file)
    assert os.path.isfile(im_file)

    return im_file, kp_file, desc_file

def match_images(file1, file2, min_match_dist = None, verbose = True, points_dir = None):

    def printv(string):
        if verbose:
            print(string)
    printv("match_images -- verbose mode")

    assert(os.path.isfile(file1))
    assert(os.path.isfile(file2))
    
    results = {}

    #load images and kp
    desc_dir = 'matched_descriptors'
    kp_dir = 'matched_keypoints'
    
    im_file1, kp_file1, desc_file1 = get_files(file1, desc_dir=desc_dir, kp_dir=kp_dir)
    im_file2, kp_file2, desc_file2 = get_files(file2, desc_dir=desc_dir, kp_dir=kp_dir)
    results["im_file1"], results["kp_file1"], results["desc_file1"] = im_file1, kp_file1, desc_file1
    results["im_file2"], results["kp_file2"], results["desc_file2"] = im_file2, kp_file2, desc_file2
    
    im1 = cv2.imread(im_file1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im_file2, cv2.IMREAD_COLOR)
    results["im1"], results["im2"] = im1, im2

    kp1, desc1 = readKpFile(kp_file1), readDescFile(desc_file1)
    kp2, desc2 = readKpFile(kp_file2), readDescFile(desc_file2)

    kp1_un, kp2_un = undistort_keypoints(kp1), undistort_keypoints(kp2)
    results["kp1"], results["desc1"], results["kp1_un"] = kp1, desc1, kp1_un
    results["kp2"], results["desc2"], results["kp2_un"] = kp2, desc2, kp2_un
    
    printv("{} total points".format(len(kp1_un)))

    m_kp1, m_kp2, matches = match_descriptors(kp1_un, desc1, kp2_un, desc2)
    printv("{} matches before ransac".format(len(matches)))
    
    pix_dist = get_pixel_distances(kp1, kp2, matches)
    match_dist = get_match_distances(desc1, desc2, matches)
    results["match_dist"] = match_dist
    results["min_match_dist"] = min_match_dist
    printv("Average Pixel Distance Between Matches: {}".format(pix_dist.mean()))
    printv("Average Descriptor Distance Between Matches: {}".format(match_dist.mean()))
    if not min_match_dist is None:
        print("filtering out matches with dist greater than {}".format(min_match_dist))
        matches = [m for m, d in zip(matches,match_dist.tolist()) if d < min_match_dist]
        m_kp1 = [k for k, d in zip(m_kp1,match_dist.tolist()) if d < min_match_dist]
        m_kp2 = [k for k, d in zip(m_kp2,match_dist.tolist()) if d < min_match_dist]
        if len(matches) < 3:
            printv("Less than 3 matches below threshold {}".format(min_match_dist))
            return results
    results["matches"] = matches
    results["match_tuples"] = [(m.queryIdx, m.trainIdx) for m in matches]
        
    # H, inliers = compute_homography(m_kp1, m_kp2)
    # printv("Matches from homography {}".format(np.sum(inliers)))
    H, inliers = compute_fundamental(m_kp1, m_kp2)
    printv("Matches from fundamental {}".format(np.sum(inliers)))
    E = K.T @ H @ K
    R1, R2, t = cv2.decomposeEssentialMat(E)
    # printv("R1 = {}".format(R1))
    # printv("R2 = {}".format(R2))
    # printv("t = {}".format(t))
    R1 = R.from_matrix(R1)
    R2 = R.from_matrix(R2)
    # printv()
    # printv(R1.as_euler('zyx'))
    # printv(R2.as_euler('zyx'))
    results['inliers'] = inliers


    im_kp1 = cv2.drawKeypoints(im1, kp1, None) #cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    im_kp2 = cv2.drawKeypoints(im2, kp2, None)
    im_kp = np.concatenate((im_kp1, im_kp2), axis = 1)
    results['im_kp'] = im_kp
    if verbose:
        matched_img = cv2.drawMatches(im1, kp1, im2, kp2, matches,
                                        None, matchColor=(0, 255, 0),
                                        singlePointColor=(0, 0, 255))
        results['matched_img'] = matched_img
    inlier_matches = np.array(matches)[inliers.astype(bool)].tolist()
    results["inlier_matches"] = inlier_matches
    results["inlier_match_tuples"] = [(m.queryIdx, m.trainIdx) for m in inlier_matches]

    pix_dist = get_pixel_distances(kp1, kp2, inlier_matches)
    match_dist_ransac = get_match_distances(desc1, desc2, inlier_matches)
    # printv("Average distance after findamental matrix rejection")
    printv("Average Pixel Distance Between Matches afer ransac: {}".format(pix_dist.mean()))
    printv("Average Distance Between Matches after ransac: {}".format(match_dist_ransac.mean()))
    results['match_dist_ransac'] = match_dist_ransac
    
    if verbose:
        matched_img_fundamental = cv2.drawMatches(im1, kp1, im2, kp2, inlier_matches,
                                        None, matchColor=(0, 255, 0),
                                        singlePointColor=(0, 0, 255))
        combo_image = np.column_stack((matched_img,matched_img_fundamental))
        cv2.imshow("test", combo_image)
        cv2.waitKey(0)
        results["matched_img_fundamental"] = matched_img_fundamental
    # match_mosaic = np.concatenate((im_kp, matched_img, matched_img_fundamental), axis = 0)
    return results

def match_confidence(file1, file2, match_threshold = 20):
    '''Tries a brute force match between saved descriptors to reject image pairs that do not correlate well
    threshold requires a certain number of matched kps to pass a fundamental matrix check'''

    result = match_images(file1, file2, verbose = False)
    return np.sum(result["inliers"])


if __name__ == "__main__":
    # desc_file_1 = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/matched_descriptors/1689804908243000031.npy"
    desc_file_1 = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/matched_descriptors/1689804910375999928.npy"
    desc_file_2 = "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/gluestick/matched_descriptors/1689819795640000105.npy"
    match_images(desc_file_1, desc_file_2, verbose = True)

    print(match_confidence(desc_file_1, desc_file_2))

