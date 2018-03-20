#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>
#include <unistd.h>
#include <sys/socket.h>                                                                                       
#include <sys/un.h>                                                                                           
#include <unistd.h>                                                                                           
#include <string.h> 
#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
image get_image_from_stream(CvCapture *cap);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier_thresh = .5;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static float *avg;

static int dx;
static int dy;
static int w;
static int h;


////CLIENT SETUP
static  struct sockaddr_un address;                                                                                  
static  int  socket_fd, nbytes;                                                                                      
//static  char result[1600];
//static  char total[640*480*3+1600]; 
//////////

void img_print(image foo){
	int i;
	char * new_str ; 
	for (i=0;i <foo.w*foo.h*foo.c;i++) {
		printf("%lf ",foo.data[i]);
	}
}

void *fetch_in_thread(void *ptr)
{
    image tmp = get_image_from_stream(cap);
    if(!tmp.data){
        error("Stream closed.");
    }
	image crop = crop_image(tmp, dx, dy, w, h);
	in = resize_image(crop, 640, 480);
    in_s = resize_image(crop, net.w, net.h);
	free_image(tmp);
	free_image(crop);
    return 0;
}

void *detect_in_thread(void *ptr)
{
	char result[1600];
	memset(result, 0, sizeof(result));
	char total[640*480*3+1600];
	memset(total, 0, sizeof(total));
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0, demo_hier_thresh);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    images[demo_index] = det;
    det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
    demo_index = (demo_index + 1)%FRAMES;
	
//concat the bbox
	int i;                         
    memset(result, '\0', sizeof(result));                                                                     
                                                                         
    int count = 0;
	int num = l.w*l.h*l.n;                                                                                            
    for(i = 0; i < num; ++i){                                                                                 
        int class = max_index(probs[i], demo_classes);                                                             
        float prob = probs[i][class];                                                                         
        if(prob > 0.2){                                                                                       
            char buffer[80];
			memset(buffer, '\0', sizeof(buffer));                                                                                  
            box b = boxes[i];                                                                                 
            snprintf(buffer, sizeof buffer, "%.0f %.0f %.0f %.0f %s %.0f%%\n", b.x*640, b.y*480, b.w*640, b.h*480, demo_names[class], prob*100);          
            strcat(result, buffer);                                                                           
            count++;                                                                                          
        }                                                                                                     
    }                                                                                                         
    strcat(result, "2\nended\n");
	strcat(total, result);
//concat the pil
    image copy = copy_image(det);                                                                               
    if(det.c == 3) 1+1 ;// rgbgr_image(copy);                                                                           
    int x,y,k;                                                                                                
                                                                                                            
    IplImage *disp = cvCreateImage(cvSize(det.w, det.h), IPL_DEPTH_8U, det.c);                                       
    int step = disp->widthStep;                                                                               
    for(y = 0; y < det.h; ++y){                                                                                 
        for(x = 0; x < det.w; ++x){                                                                             
            for(k= 0; k < det.c; ++k){                                                                          
                disp->imageData[y*step + x*det.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*254 + 1);             
            }                                                                                                 
        }                                                                                                     
    }
	strcat(total, disp->imageData);                                                                           
    free_image(copy);  
	printf("%i", write(socket_fd, total, sizeof(total)));
	memset(total, 0, sizeof(total));
    cvReleaseImage(&disp);                                                                                    
    return 0;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh,char *socket_param)// CLEAN_UP
{
	int zoom[4];
	printf("bbox %s", prefix);
	char *token;
	int i =0;
	while ((token = strsep(&prefix, "x"))){
		zoom[i] =atoi( token);
		i++;
	} 
	dx = zoom[0];
	dy = zoom[1];
	w = zoom[2];
	h = zoom[3];
    socket_fd = socket(PF_UNIX, SOCK_STREAM, 0);                                                                 
    if(socket_fd < 0)                                                                                            
    {                                                                                                            
        printf("socket() failed\n");                                                                                
		exit(0);                                                                                                   
    }

    memset(&address, 0, sizeof(struct sockaddr_un));                                                             
                                                                                                              
    address.sun_family = AF_UNIX;   
    //CLEAN_UP [START]                                                        
    snprintf(address.sun_path, 30 , socket_param); 
    //snprintf(address.sun_path, 30 , "/tmp/DockerPipes/sss"); 
    //CLEAN_UP [END]
    if(connect(socket_fd,                                                                                        
		(struct sockaddr *) &address,                                                                     
        sizeof(struct sockaddr_un)) != 0)                                                                 
    {                                                                                                            
        printf("connect() failed\n");                                                                               
    return 1;                                                                                                   
    } 
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier_thresh = hier_thresh;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    for(j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
        det = in;
        det_s = in_s;
    }

    int count = 1;

    double before = get_wall_time();

    while(count){
//        sleep(0.5);
        ++count;

            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

            if(delay == 0){
                free_image(disp);
                disp  = det;
            }
            det   = in;
            det_s = in_s;

        --delay;
        if(delay < 0){
            delay = frame_skip;

            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            before = after;
        }
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh,char *socket_param)// CLEAN_UP
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

