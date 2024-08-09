import numpy as np 
import cv2
width=512
height=256
ch=19

colorB = [128, 232, 70, 156, 153, 153, 30,  0,   35, 152, 180, 60,  0,  142, 70,  100, 100, 230, 32]
colorG = [64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130, 20, 0,  0,   0,   60,  80,  0,   11]
colorR = [128, 244, 70,  102, 190, 153, 250, 220, 107, 152,70,  220, 255, 0,   0,   0,   0,   0,   119]


arr_tm = np.loadtxt("mask_256.csv", delimiter=",").astype(np.int64) # csv file containing the class of each pixel of the truth mask
arr_f=np.loadtxt("class_faulty_0_new.csv", delimiter=",",dtype=int) # csv file containing the class of each pixel of the faulty mask



img_ind_fault=arr_f.reshape(height,width)
img_ind_tm = arr_tm[0,:].reshape(height,width)

#img_ind_fault = np.argmax(img_fault, axis=2) 

IoU_arr = []  # array to store the IoU of each class (19 elements)

for max_class in range(0, 19):

#If an index contains the class we are caculating the IoU for, then it will be convereted to 1. Otherwise it will converted to 0.
# This happens for each of the 19 classes to calculate the IoU
    
    faulty_img = np.where(img_ind_fault == max_class, 1, 0) 
    truth_mask_img = np.where(img_ind_tm == max_class, 1, 0) 

    tp = 0
    fp = 0
    fn = 0
    tn = 0
# The number of true positives, faulse positives etc are calculated
    for i in range(faulty_img.shape[0]):
        for j in range(faulty_img.shape[1]):
            if  faulty_img[i, j]== 1 and truth_mask_img[i, j] == 1:
                tp += 1
            elif faulty_img[i, j] == 0 and truth_mask_img[i, j] == 0:
                tn += 1   
            elif faulty_img[i, j] > truth_mask_img[i, j]:
                fp += 1
            elif faulty_img[i, j] < truth_mask_img[i, j]:
                fn += 1
    # Print the results
    print(f"class {max_class}")
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
# If number of true positives is zero, IoU is not defined
    if tp == 0:
        IoU = 0
        print(f"IoU: {IoU}")
        print("\n")
        IoU_arr.append(IoU)
    else:
        IoU = tp/(tp+fp+fn) # formula for IoU
        print(f"IoU: {IoU}")
        print("\n")
        IoU_arr.append(IoU)

print(IoU_arr) // It contains the IoUs of each class.

# Computing and printing the mean IoU taking into account the classes whose IoU is non zero (defined)
sum_of_IoU = np.sum(IoU_arr)
non_zero_count = np.count_nonzero(IoU_arr)
mean_IoU = sum_of_IoU/non_zero_count
print("\n")
print(f"Mean IoU: {mean_IoU}")
print("\n")
