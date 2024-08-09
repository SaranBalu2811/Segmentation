# Segmentation
1. **my_final_8_fpn_glitch.cc** - Used to inject faults at 8 different offsets and generate output image

            File used to build the code: build_my_final_8_fpn_glitch.sh
            command to build: ./build_my_final_8_fpn_glitch.sh
            Command to run the code: ./segmentation /usr/share/vitis_ai_library/models/fpn/fpn.xmodel input.png 4000 48200 11
             where: argument 1: input image (input.png)
                    argument 2: offset where the glitch should be injected (4000 * 100)
                    argument 3: Maximum offset (48200 * 100)
                    argument 4: Width of the glitch (11)
            The ouput will be test.png (fault-free output) and test-faulty.png (faulty).
   
2. **my_final_8_fpn_glitch_search.cc** - Used to inject faults at 8 different offsets (with a step size) and outputs the pixel difference.

                  File used to build the code: build_my_final_8_fpn_glitch_search.sh
                  command to build: ./build_my_final_8_fpn_glitch_search.sh
                  Command to run the code: ./segmentation /usr/share/vitis_ai_library/models/fpn/fpn.xmodel input.png 4000 20000 11
                         where: argument 1: input image (input.png)
                                argument 2 and 3: Range of offsets in which the glitch should be injected (Step size should be specified in the code).
                                argument 4: Width of the glitch (11).
                 The ouput will be offset, pixel difference ,inference time.
   
   
3. **fpn_glitch_search_end_pixels_test.cc** - Used to attack the last layer and it generates output only if the (inference time - offsest) is between 2000 to 4000 (can be changed). the output image file contains the offset, inference, their difference and the pixel value.

                  File used to build the code: build_fpn_glitch_search_end_pixels_test.sh
                  command to build: ./build_fpn_glitch_search_end_pixels_test.sh
                  Command to run the code: ./segmentation /usr/share/vitis_ai_library/models/fpn/fpn.xmodel input.png 47000 48520 2
                         where: argument 1: input image (input.png)
                                argument 2 and 3: Range of offsets in which the glitch should be injected (Step size should be specified in the code).
                                argument 4: Width of the glitch (2).
                 The ouput will be faulty images with image file containing the offset, inference, their difference and the pixel value.
   
4. **fpn_glitch_pix_val.cc** - Used to generate a csv file of pixel values of size 256*512*19 for both fault-free and faulty mask. 

               File used to build the code: build_fpn_glitch_pix_val.sh
                  command to build: ./build_fpn_glitch_pix_val.sh
                  Command to run the code: ./segmentation /usr/share/vitis_ai_library/models/fpn/fpn.xmodel input.png 4000 48200 11
                         where: argument 1: input image (input.png)
                                argument 2: offset where the glitch should be injected (4000 * 100)
                                argument 3: Maximum offset (48200 * 100)
                                argument 4: Width of the glitch (11).
                 The ouput will be a csv file of pixel values of size 256*512*19 for both fault-free and faulty mask and it will stored in "scp /run/media/sda1/seg_records/".
                 Command to copy the code: scp /run/media/sda1/seg_records/fpn_rand_outputs_4000_48200_11.csv twh-lab@192.168.1.101:~/Desktop
   
5. **test_my_seg_output.py** - The indexes of the pixels whose values (faulty values) differ significantly from the original values can been recorded.

         Command to run: python3 test_my_seg_output.py
         The input to this code is the csv file containing the pixel value that was generated using the previous code.
         The ouput will be index along with the difference between the original max value and faulty max value.
   
6. **IoU_org_new.py** - Calculate the IoU of each classes between the fault-free and ground truth mask. The input should be an array (can be uploaded as csv file) of class of each index.

       Command to run: python3 IoU_org_new.py
       Input 1: csv file containing the class of each pixels of fault-free mask (size = 256*512).
       Input 2: csv file containing the class of each pixels of truth mask (size = 256*512).
       Output will be IoU of each class.
   
7. **IoU_flt_new.py** - Calculate the IoU of each classes between the faulty and ground truth mask. The input should be an array (can be uploaded as csv file) of class of each index.

                Same as the privious code. 
8. **fpn_mob_glitch.cc** - For Mobilenet
9. **fpn_mob_glitch_search.cc** - For Mobilenet
10. **fpn_mob_glitch_search_end_pixels.cc** - For MobileNet
