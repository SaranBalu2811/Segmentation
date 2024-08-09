import numpy as np 
import cv2
width=512
height=256
ch=19

colorB = [128, 232, 70, 156, 153, 153, 30,  0,   35, 152, 180, 60,  0,  142, 70,  100, 100, 230, 32]
colorG = [64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130, 20, 0,  0,   0,   60,  80,  0,   11]
colorR = [128, 244, 70,  102, 190, 153, 250, 220, 107, 152,70,  220, 255, 0,   0,   0,   0,   0,   119]

arr = np.loadtxt("fpn_rand_outputs_end_pixels_4848930_4820000_2.csv",delimiter=",",dtype=int) # load the csv file with pixel values
arr=arr[:,:-1]


img=arr[0,:].reshape(height,width,ch)
img_fault=arr[1,:].reshape(height,width,ch)

img_ind=np.argmax(img, axis=2) # select the index of max value along the channel axis (among 19 values).


img_ind_fault = np.argmax(img_fault, axis=2) 


rows, cols = np.indices(img_ind.shape) 

max_values_o = img[rows, cols, img_ind] #using the index value found above, get the max value at that index
#print(max_values_o)

max_values_f = img_fault[rows, cols, img_ind_fault]
#print(max_values_f)

error = max_values_f - max_values_o # Difference between the faulty and original value

print(error)
print("\n")

print("Max error value is " )
print(error.max())
print("\n")

# Print the index of image where the differnt between original max value and faulty max value is greater than 5.
print("\nError values > 5")
for row_idx in range(img_ind.shape[0]):  # number of rows
    for col_idx in range(img_ind.shape[1]):  # number of columns
        if error[row_idx, col_idx]  > 5:
            idx = (row_idx, col_idx, img_ind_fault[row_idx, col_idx])
            fault_val = max_values_f[row_idx, col_idx]
            original_val = max_values_o[row_idx, col_idx]
            print(f"Index: {idx}, Faulty_Max_Value: {fault_val}, Original_Max_Value: {original_val}, Difference: {error[row_idx, col_idx]}")


print("\nError values < -5")
for row_idx in range(img_ind.shape[0]):  # number of rows
    for col_idx in range(img_ind.shape[1]):  # number of columns
        if error[row_idx, col_idx]  < -5:
            idx = (row_idx, col_idx, img_ind_fault[row_idx, col_idx])
            fault_val = max_values_f[row_idx, col_idx]
            original_val = max_values_o[row_idx, col_idx]
            print(f"Index: {idx}, Faulty_Max_Value: {fault_val}, Original_Max_Value: {original_val}, Difference: {error[row_idx, col_idx]}")


print(np.max(img))
print(np.min(img))

print(np.max(img_fault))
print(np.min(img_fault))

img_mask=np.zeros((height,width,3))#.astype(int)
for i in range(img_ind.shape[0]):
	for j in range(img_ind.shape[1]):
		img_mask[i,j,0]=colorB[img_ind_fault[i,j]]
		img_mask[i,j,1]=colorG[img_ind_fault[i,j]]
		img_mask[i,j,2]=colorR[img_ind_fault[i,j]]
		
img_mask=img_mask.astype(np.uint8)
cv2.imshow("mask",img_mask ) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

print(arr.shape)
print(img.shape)
print(img_ind.shape)
print(img_mask)
