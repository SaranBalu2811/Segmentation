#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "common.h"
#include <fcntl.h>
#include <sys/mman.h>
GraphInfo shapes;
using namespace std;
using namespace std::chrono;
using namespace cv;
unsigned int axi_size = 0x10000;
off_t axi_pbase = 0xa0000000; /* physical base address */
off_t s_axi_base= 0x8f000000;
uint *axi_vptr;
uint *s_axi_ptr;
int fd;
bool ground_flag=true;
unsigned int offset;
unsigned int min_offset;
unsigned int max_offset;
unsigned int num_glitches=2;
unsigned int width;
unsigned int step=500;
unsigned int glitch_step=25;
//int start_offset = 400000;
//int offset = 400050;
int i=0;
unsigned int limit;

//unsigned int off_a; 
//unsigned int off_b; //[3] = {400000, 401000, 402000};
//unsigned int off_c; //[3] = {670000, 671000, 672000};

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
                    180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64,  35, 70, 102, 153, 153, 170, 220, 142, 251,
                    130, 20, 0,  0,   0,   60,  80,  0,   11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
                    70,  220, 255, 0,   0,   0,   0,   0,   119};

// comparison algorithm for priority_queue
class Compare {
 public:
  bool operator()(const pair<int, Mat>& n1, const pair<int, Mat>& n2) const {
    return n1.first > n2.first;
  }
};

// input video
VideoCapture video;
//const string img_path="input.png";
string img_path;
// flags for each thread
bool is_reading = false;
bool is_running_1 = true;
bool is_running_2 = false;
bool is_displaying = false;

queue<pair<int, Mat>> read_queue;  // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare>
    display_queue;        // display queue
mutex mtx_read_queue;     // mutex of read queue
mutex mtx_display_queue;  // mutex of display queue
int read_index = 0;       // frame index of input video
int display_index = 0;    // frame index to display

/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param task - pointer to Segmentation Task
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runSegmentation(vart::Runner* runner, bool& is_running) {
  // init out data
  float mean[3] = {104, 117, 123};
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
  int batch = inputTensors[0]->get_shape().at(0);
  int8_t* result = new int8_t[shapes.outTensorList[0].size * batch];
  int8_t* imageInputs = new int8_t[shapes.inTensorList[0].size * batch];

  /* ----------- Glitch parameters setup ---------------*/
  /* set pointer for axi */
  if ((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) {
    printf("Access memory error");
    //return(0);
  }
  axi_vptr = (uint *)mmap(NULL, axi_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, axi_pbase);
  s_axi_ptr =   (uint *)mmap(NULL, axi_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, s_axi_base);

  while (is_running) {
    // Get an image from read queue
    int index;
    Mat img;
    img = imread(img_path, IMREAD_COLOR);

    // get in/out tensor
    auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());
    auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
    auto input_scale = get_input_scale(runner->get_input_tensors()[0]);

    // get tensor shape info
    int outHeight = shapes.outTensorList[0].height;
    int outWidth = shapes.outTensorList[0].width;
    int inHeight = shapes.inTensorList[0].height;
    int inWidth = shapes.inTensorList[0].width;

    // image pre-process
    Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
    resize(img, image2, Size(inWidth, inHeight), 0, 0, INTER_LINEAR);

    for (int h = 0; h < inHeight; h++) {
      for (int w = 0; w < inWidth; w++) {
        for (int c = 0; c < 3; c++) {
          imageInputs[h * inWidth * 3 + w * 3 + c] = (int8_t)(
              ((float)image2.at<Vec3b>(h, w)[c] - mean[c]) * input_scale);
        }
      }
    }

    // tensor buffer prepare
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, inputTensors[0].get()));
    outputs.push_back(
        std::make_unique<CpuFlatTensorBuffer>(result, outputTensors[0].get()));

    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    bool f_csv;
    /*run*/
    ground_flag = true;
    offset= min_offset;        
    for (int m=0; m<limit ; m++){
            
                for (int a=0; a<num_glitches; a++){
                    axi_vptr[2*a] = 108000 + a*glitch_step;
                    axi_vptr[2*a+1] = 108000 + a*glitch_step+ width;
                  
                }

                for (int c=0; c<num_glitches; c++){
                    axi_vptr[(2*num_glitches)+(2*c)] = 412500 + c*glitch_step;
                    axi_vptr[(2*num_glitches)+(2*c+1)] = 412500 + c*glitch_step+ width;
                  
                }

                for (int c=0; c<num_glitches; c++){
                    axi_vptr[(4*num_glitches)+(2*c)] = 521000 + c*glitch_step;
                    axi_vptr[(4*num_glitches)+(2*c)+1] = 521000 + c*glitch_step + width;
                }

                for (int d=0; d<num_glitches; d++){
                    axi_vptr[(6*num_glitches)+(2*d)] = 708000 + d*glitch_step;
                    axi_vptr[(6*num_glitches)+(2*d)+1] = 708000 + d*glitch_step + width;
                }

                for (int e=0; e<num_glitches; e++){
                    axi_vptr[(8*num_glitches)+(2*e)] = 827500 + e*glitch_step;
                    axi_vptr[(8*num_glitches)+(2*e)+1] = 827500 + e*glitch_step + width;
                }

                for (int f=0; f<num_glitches; f++){
                    axi_vptr[(10*num_glitches)+(2*f)] = 1496000+ f*glitch_step;
                    axi_vptr[(10*num_glitches)+(2*f)+1] = 1496000 + f*glitch_step + width;
                }
                

                for (int g=0; g<num_glitches; g++){
                    axi_vptr[(12*num_glitches)+(2*g)] = 2423000+ g*glitch_step;
                    axi_vptr[(12*num_glitches)+(2*g)+1] = 2423000 + g*glitch_step + width;
                }

                for (int h=0; h<num_glitches; h++){
                    axi_vptr[(14*num_glitches)+(2*h)] = offset+ m*step + h*glitch_step;
                    axi_vptr[(14*num_glitches)+(2*h)+1] = offset + m*step + h*glitch_step + width;
                }

                for(int ii=num_glitches*16;ii<198;ii++)
                {
                    axi_vptr[ii]=offset+num_glitches*glitch_step;
                }
                
                
                      
                      
                    for (unsigned int j = 0; j < 2; j++) {
                            if (j % 2 == 0) {
                            int status = system("echo 0 > /sys/class/gpio/gpio78/value");
                            f_csv = false;
                            } else {
                            int status = system("echo 1 > /sys/class/gpio/gpio78/value");
                            f_csv = true;
                            ground_flag = false;
                            }
                            auto job_id = runner->execute_async(inputsPtr, outputsPtr);
                        runner->wait(job_id.first, -1);
                        //printf("%u-%u\n",axi_vptr[198],axi_vptr[199]);
                        Mat segMat(outHeight, outWidth, CV_8UC3);
                        Mat segMat_free(outHeight, outWidth, CV_8UC3);
                        Mat showMat(inHeight, inWidth, CV_8UC3);
                        for (int row = 0; row < outHeight; row++) {
                            for (int col = 0; col < outWidth; col++) {
                            int i = row * outWidth * 19 + col * 19;
                            auto max_ind = max_element(result + i, result + i + 19);
                            int posit = distance(result + i, max_ind);
                            segMat.at<Vec3b>(row, col) =
                                Vec3b(colorB[posit], colorG[posit], colorR[posit]);
                            if(j % 2 == 0) segMat_free.at<Vec3b>(row, col) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);
                            }
                        }

                        // resize to original scale and overlay for displaying
                        resize(segMat, showMat, Size(inWidth, inHeight), 0, 0, INTER_NEAREST);
                        for (int i = 0; i < showMat.rows * showMat.cols * 3; i++) {
                            img.data[i] = img.data[i] * 0.4 + showMat.data[i] * 0.6;
                        }
                        if(j % 2 == 0)  
                        {
                            //imwrite("test.png", showMat);
                            //segMat_free=segMat;
                        }
                        if(j % 2 == 1) {
                            //imwrite("test-faulty.png", showMat);
                            int diff_pixel=0;
                            //Vec3b colorx = segMat_free.at<Vec3b>(5, 5);
                            
                            for (int row = 0; row < outHeight; row++) {
                            for (int col = 0; col < outWidth; col++) {
                                if((segMat.at<Vec3b>(row, col)[0]!=segMat_free.at<Vec3b>(row, col)[0]) || (segMat.at<Vec3b>(row, col)[1]!=segMat_free.at<Vec3b>(row, col)[1])||(segMat.at<Vec3b>(row, col)[2]!=segMat_free.at<Vec3b>(row, col)[2])) diff_pixel++;
                                //if(segMat.at<Vec3b>(row, col)!=segMat_free.at<Vec3b>(row, col)) diff_pixel++;
                            }
                            }
                            //printf("Offset : %u    Diff pixels:%d  inference cycles: %u\n",offset+k*step,diff_pixel,axi_vptr[199]);

                            if (diff_pixel >= 3300)
                            printf("%u,%u,%u,%u,%u,%u,%u,%u,%d,%u\n", 108000, 412500, 521000, 708000, 827500, 1496000,2423000, offset+m*step, diff_pixel,axi_vptr[199]);
                            //printf("Diff pixels:%u,%u,%u\n",colorx[0],colorx[1],colorx[2]);
                        }

                    }
            
        
    }
      /* Setting glitch offset and width*/
      /* integers number are just a reference */
      
        
       /*for(int k = 12; k <198; k++)
        {
            axi_vptr[k]= offset;
            offset+=25;
            
        }
*/ 


      // run
      
    
    
    inputsPtr.clear();
    outputsPtr.clear();
    inputs.clear();
    outputs.clear();
    is_running = false;
  }

  delete imageInputs;
  delete result;
}


/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool& is_reading) {
  while (is_reading) {
    Mat img;
    if (read_queue.size() < 30) {
      if (!video.read(img)) {
        cout << "Finish reading the video." << endl;
        is_reading = false;
        break;
      }
      mtx_read_queue.lock();
      read_queue.push(make_pair(read_index++, img));
      mtx_read_queue.unlock();
    } else {
      usleep(20);
    }
  }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool& is_displaying) {
  while (is_displaying) {
    mtx_display_queue.lock();
    if (display_queue.empty()) {
      if (is_running_1 || is_running_2) {
        mtx_display_queue.unlock();
        usleep(20);
      } else {
        is_displaying = false;
        break;
      }
    } else if (display_index == display_queue.top().first) {
      // Display image
      imshow("Segmentaion @Xilinx DPU", display_queue.top().second);
      display_index++;
      display_queue.pop();
      mtx_display_queue.unlock();
      if (waitKey(1) == 'q') {
        is_reading = false;
        is_running_1 = false;
        is_running_2 = false;
        is_displaying = false;
        break;
      }
    } else {
      mtx_display_queue.unlock();
    }
  }
}

/**
 * @brief Entry for running Segmentation neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char** argv) {
  // Check args
  if (argc != 6) {
        cout << "Usage: " << argv[0] << " <model_name>" << " <image_path>" << " <min_offset>"<<" <max_offset>" <<" <model FC>" <<" <max width>"<< endl;
        return -1;
  }

  min_offset =atoi(argv[3])*100;
  max_offset =atoi(argv[4])*100;
  //off_c =atoi(argv[5])*100;
  limit=(max_offset-min_offset)/step;

  width=atoi(argv[5]);
  // Initializations
  
  //string file_name = argv[1];
  img_path=argv[2];
  /*
  cout << "Detect video: " << file_name << endl;
  video.open(file_name);
  if (!video.isOpened()) {
    cout << "Failed to open video: " << file_name;
    return -1;
  }

  */
  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "segmentation should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  // create runner
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  // in/out tensors
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();

  // get in/out tensor shape
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  // Run tasks
  array<thread, 4> threads = {
      thread(Read, ref(is_reading)),
      thread(runSegmentation, runner.get(), ref(is_running_1)),
      thread(runSegmentation, runner2.get(), ref(is_running_2)),
      thread(Display, ref(is_displaying))};

  for (int i = 0; i < 4; ++i) {
    threads[i].join();
  }

  //video.release();

  return 0;
}

