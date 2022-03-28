# Hybrid Tracker
A Hybrid Approach for Tracking Individual Players in Broadcast Match Videos

## Build the project

1. Set up a Tensorflow baseline project

* Clone the full Tensorflow object detection repository located at ``` https://github.com/tensorflow/models```

```
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
```

* Download the [Faster-RCNN-Inception-V2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) and the [SSD_MOBILENET-V1_COCO](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz) models into the ```tensorflow/models/research/object_detection/``` folder.

2. Configure HybridTracker

* Clone this repository and move the content of the ```object_detection``` folder to ```tensorflow/models/research/object_detection/```

```
git clone git@github.com:LopezCastroRoberto/HybridTracker.git
mv HybridTracker/object_detection/* tensorflow/models/research/object_detection/
``` 

3. Create a new Anaconda Virtual Environment
```
conda create -n tensorflow
source activate tensorflow
python -m pip install --upgrade pip
python -m pip install --ignore-installed --upgrade tensorflow-gpu
conda install -c anaconda protobuf
python -m pip install -r requirements.txt
```

4. Compile protobufs and run setup.py
```
cd tensorflow/models/research
export PYTHONPATH=$PYTHONPATH:'pwd':'pwd/slim'
```

```
protoc --python_out=. ./object_detection/protos/anchor_generator.proto ./object_detection/protos/argmax_matcher.proto ./object_detection/protos/bipartite_matcher.proto ./object_detection/protos/box_coder.proto ./object_detection/protos/box_predictor.proto ./object_detection/protos/eval.proto ./object_detection/protos/faster_rcnn.proto ./object_detection/protos/faster_rcnn_box_coder.proto ./object_detection/protos/grid_anchor_generator.proto ./object_detection/protos/hyperparams.proto ./object_detection/protos/image_resizer.proto ./object_detection/protos/input_reader.proto ./object_detection/protos/losses.proto ./object_detection/protos/matcher.proto ./object_detection/protos/mean_stddev_box_coder.proto ./object_detection/protos/model.proto ./object_detection/protos/optimizer.proto ./object_detection/protos/pipeline.proto ./object_detection/protos/post_processing.proto ./object_detection/protos/preprocessor.proto ./object_detection/protos/region_similarity_calculator.proto ./object_detection/protos/square_box_coder.proto ./object_detection/protos/ssd.proto ./object_detection/protos/ssd_anchor_generator.proto ./object_detection/protos/string_int_label_map.proto ./object_detection/protos/train.proto ./object_detection/protos/keypoint_box_coder.proto ./object_detection/protos/multiscale_anchor_generator.proto ./object_detection/protos/graph_rewriter.proto ./object_detection/protos/calibration.proto ./object_detection/protos/flexible_grid_anchor_generator.proto
```

```
python setup.py build
python setup.py install
```

**Note: the Tensorflow models repository is in continuous development what can break the functionality of this brief tutorial. Check out the new updates of the Tensorflow models repository in order to adapt this setup as needed. This tutorial was originally done using TensorFlow 1.13.1 and more specifically this [GitHub commit](https://github.com/tensorflow/models/tree/0b0dc7f546f38a34d9f49279a1ad899e1a46b8b6) of the Tensorflow Object Detection API.**
## Run
### Training
1. Create an empty folder called ```images``` with two sub-folders: ```train``` and ```test```.
2. [Label your images](https://github.com/tzutalin/labelImg) or divide your already labeled ones into ```train``` and ```test``` folders. A common strategy is to put 80% of them into the ```train``` directory and the rest to the ```test```one. Your labeled data must be placed outside ```train``` and ```test```, in ```images```.
2.1. If your labeled data is in ```.xml``` format it must be translated to ```.csv```. Change directory to ```object detection``` and execute:
```
python xml_to_csv.py
```
to translate xml labels to csv format.

3. Modify ```generate_tfrecord.py``` file specifying the classes labeled in our data set.
4. Execute
```
python generate_tfrecord.py --csv_input=images/train_label.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_label.csv --image_dir=images/test --output_path=test.record
```
5. Create a file called ```labelmap.pbtxt``` inside ```object_detection/training``` folder specifying the classes we are going to use.
6. Copy your models configuration files from ```object_detection/samples/config``` to ```training``` directory. In our case, we need to create a configuration file to Faster-RCNN model and another one to SSD one.
7. Adapt previous model configuration file to our use-case. Modify:
* Number of classes
* ```fine_tune_checkpoint``` parameter indicating the path of the downloaded models.
* Paths to train and test directories as well as to ```labelmap.pbtxt``` file.
* Number of tests images in ```eval_config``` field.
8. Train the network. Execute:
```
python train.py --logtostderr --train_dir=training --pipeline_config_path=training/faster-rcnn_inception_v2_pats.config
```

9. Export inference graph. Select the newest checkpoint generated in ```object_detection/training``` folder and execute:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained/model.ckpt_xxxx --output_directory inference_graph
```

A ```.pb``` file is created into ```inference_graph``` directory which we will refer to on inference time.

### Inference
```
 python hybrid_tracker.py jump crop_x crop_y faster_rcnn_model_folder ssd_model_folder crop_z path_to_results path_to_video
```
for example,
```
 python hybrid_tracker.py 6 125 90 inference_graph_fasterrcnn_kross inference_graph_ssd_kross 500 paper_videos/kross_full/2_HQ2 paper_videos/kross_full/2_HQ2/2_HQ2.mp4
```

## Citation
If you find this tool helpful, please cite:

## License
Apache-2.0 License

-- Roberto López Castro
