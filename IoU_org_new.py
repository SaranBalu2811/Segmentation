import numpy as np 
import cv2
width=512
height=256
ch=19

colorB = [128, 232, 70, 156, 153, 153, 30,  0,   35, 152, 180, 60,  0,  142, 70,  100, 100, 230, 32]
colorG = [64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130, 20, 0,  0,   0,   60,  80,  0,   11]
colorR = [128, 244, 70,  102, 190, 153, 250, 220, 107, 152,70,  220, 255, 0,   0,   0,   0,   0,   119]

arr_o = np.loadtxt("class_original_0.csv", delimiter=",",dtype=int) # csv file containing the class of each pixel of the truth mask
arr_tm = np.loadtxt("mask_256.csv", delimiter=",").astype(np.int64)
#arr_o=arr_o[:,:-1]



img_ind_original = arr_o.reshape(height,width)
img_ind_tm = arr_tm[0,:].reshape(height,width)

#img_ind_orginal=np.argmax(img_original, axis=2)

IoU_arr = []

for max_class in range(0, 19):

    original_img = np.where(img_ind_original == max_class, 1, 0) 
    truth_mask_img = np.where(img_ind_tm == max_class, 1, 0) 

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            if original_img[i, j] == 1 and truth_mask_img[i, j] == 1:
                tp += 1
            elif original_img[i, j] == 0 and truth_mask_img[i, j] == 0:
                tn += 1   
            elif original_img[i, j] > truth_mask_img[i, j]:
                fp += 1
            elif original_img[i, j] < truth_mask_img[i, j]:
                fn += 1

    # Print the results
    print(f"class {max_class}")
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")

    if tp == 0:
        IoU = 0
        print(f"IoU: {IoU}")
        print("\n")
        IoU_arr.append(IoU)
    else:
        IoU = tp/(tp+fp+fn)
        print(f"IoU: {IoU}")
        print("\n")
        IoU_arr.append(IoU)

print(IoU_arr)

sum_of_IoU = np.sum(IoU_arr)
non_zero_count = np.count_nonzero(IoU_arr)
mean_IoU = sum_of_IoU/non_zero_count
print("\n")
print(f"Mean IoU: {mean_IoU}")
print("\n")
