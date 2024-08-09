# Segmentation
1. my_final_8_fpn_glitch.cc - Used to inject faults at 8 different offsets and generate output image
2. my_final_8_fpn_glitch_search.cc - Used to inject faults at 8 different offsets (with a step size) and outputs the pixel difference.
3. fpn_glitch_search_end_pixels_test.cc - Used to attack the last layer and it generates output only if the (inference time - offsest) is between 2000 to 4000 (can be changed). the output image file contains the offset, inference, their difference and the pixel value.
4. fpn_glitch_pix_val.cc - Used to generate a csv file of pixel values of size 256*512*19 for both fault-free and faulty mask. It has the values of each class of a particular pixel.
5. test_my_seg_output.py - The indexes of the pixels whose values (faulty values) differ significantly from the original values can been recorded.
6. IoU_org_new.py - Calculate the IoU of each classes between the fault-free and ground truth mask. The input should be an array (can be uploaded as csv file) of class of each index.
