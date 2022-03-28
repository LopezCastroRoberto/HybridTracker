## Requirements
* opencv
* opencv_contrib

## Build the project
Move trackerKCF.cpp to ```opencv_contrib/modules/tracking/src``` and rebuild opencv_contrib

Run inside this folder
```	
cmake .
make
./autolabel
```

How to interact with the labelling tool. Press key:
* ```esc```: stop tracking/exit program
* ```s```: stop video (not tracking)
* ```r```: stop video and (re)select a ROI with mouse
* ```space```: stop tracking/(re)start tracking after pressing ```r``` key
* ```j```: jump *n* frames
