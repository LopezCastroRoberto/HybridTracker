// Copyright 2022 Roberto Lopez Castro
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Disclaimer: template code for opencv usage has been obtained from https://github.com/spmallick/learnopencv/tree/master/tracking related to
// https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <string>
#include <unistd.h>

using namespace cv;
using namespace std;

#define TIME_TO_CAPTURE 20

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char **argv)
{
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};

    string trackerType = trackerTypes[2];

    Ptr<Tracker> tracker;

    if (trackerType == "BOOSTING")
        tracker = TrackerBoosting::create();
    if (trackerType == "MIL")
        tracker = TrackerMIL::create();
    if (trackerType == "KCF")
        tracker = TrackerKCF::create();
    if (trackerType == "TLD")
        tracker = TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
        tracker = TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
        tracker = TrackerGOTURN::create();

    fstream input;
    input.open("confFile.txt");
    string vid;
    int delay = 10;
    string num;
    int index = 0;
    ofstream fs;

    if(input.is_open())
    {
        getline(input, vid);
        getline(input, num);
        index = atoi(num.c_str());
        input.close();
    } else {
        cout << "***Could not open confFile.txt ...***\n";
        return -1;
    };

    cout << vid << endl;    
    VideoCapture video(vid);
    cout << "." << endl;

    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }
    
    Mat frame;
    bool ok = video.read(frame);
    
    // Define initial boundibg box
    Rect2d bbox(607, 366, 338, 330);

    // Uncomment the line below to select a different bounding box
    //bbox = selectROI(frame, false);

    // Display bounding box.
    //rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);

    
    if(!index){
        fs.open ("training.csv", ofstream::out | ofstream::app);
        fs << "filename,width,height,class,xmin_t,ymin_t,xmax_t,ymax_t" << endl;
        fs.close();
    }

    float width  = video.get(CAP_PROP_FRAME_WIDTH);
    float height = video.get(CAP_PROP_FRAME_HEIGHT);

    int iteration = TIME_TO_CAPTURE;
    int tracking = 0;
    int numFrame = 0;

    while(video.read(frame))
    {
        string str;
        stringstream ss;

        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(1);

        if((iteration == TIME_TO_CAPTURE) && tracking){
            str = "./images/training_";
            ss << index++;
            str.append(ss.str());
            str.append(".png");

            imwrite(str, frame, compression_params);

            fs.open("training.csv", ofstream::out | ofstream::app);
            str = "training_";
            str.append(ss.str());
            str.append(".png");
            fs << str << "," << width << "," << height << "," << "varane" << ",";
            fs.close();

            iteration = 0;
            cout <<  "Now !" << endl;
        }


        iteration++;
        
        if (tracking)
        {
          bool ok = tracker->update(frame, bbox);
        }

        
        if (ok && tracking)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        } else if (tracking)
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);

            // Stop tracking
            break;
        }

        // Display tracker type on frame
        //putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

        // Display FPS on frame
        //putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Display frame.
        imshow("Tracking", frame);

        int k = waitKey(1);
        if(k == 27) // Exit if ESC pressed.
        {
            break;
        } else if(k == 114){ // (Re)start tracking if 'r' pressed.

          bbox = selectROI("Tracking", frame, false);

          tracker->clear();

          tracker = TrackerKCF::create();

          tracker->init(frame, bbox);
          tracking = 1;
          iteration = TIME_TO_CAPTURE;
        } else if (k == 115){ // stop video (not tracking) is 's' is pressed

          waitKey(0);

        } else if (k == 32){ // stop tracking (only reproduce video) is 'space' is pressed

          tracker->clear();
          tracking = 0;

        } else if (k == 74){
          numFrame-=(30+iteration);
          video.set(1, numFrame);
        } else if (k == 106){ // jump n frames if 'j' is pressed
          numFrame+=(60+iteration);
          video.set(1, numFrame);
        }

        //sleep(0.5);
        //if(waitKey(delay) >= 0) break;
    }
}
