# object_detection
================================= PART 1 =================================================
============================= setup environment ==========================================
https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/

install Object_detection
Python 2.7.6 >>> tf.__version__  >> '1.4.0'
Python 3.4.3 >>> tf.__version__  >> '1.4.0'
Protobuf 2.6 >>> protoc --version
opencv 2.4.13 
***********************************************
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu

sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib

#Protobuf 2.6
Install Protobuf 2.6
wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
tar xzf protobuf-2.6.1.tar.gz
cd protobuf-2.6.1
sudo apt-get update
sudo apt-get install build-essential
sudo ./configure
sudo make
sudo make check
sudo make install 
sudo ldconfig
protoc --version

*****************************************************
#install tensorflow_objectdetection
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

mk -dir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models/tree/master/research/object_detection

cd tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.

cd tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
***./brachc
cd From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ~/

Testing the Installation
python object_detection/builders/model_builder_test.py

***********************************************************
#example object_detection API (dog,kite)
cd ~/tensorflow/models/research/object_detection
jupyter notebook
>> object_detection_tutorial.ipynb >> cell>runall


================================= PART 2 ======================================================
============================= webcam object detaction =========================================
https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/?completed=/introduction-use-tensorflow-object-detection-api-tutorial/

# openCV
install openCV
  
  import cv2
  >>> cv2.__version__ 
  '3.2.0'
  sudo pip uninstall opencv-python
  
  2.4.13
  http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html  
[compiler] sudo apt-get install build-essential
[required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
[optional] sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

extract http://opencv.org/releases.html
cd ~/opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
*********************************************************************
#object_detection from webcam
https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/

cd tensorflow/models/research/object_detection
python my_cam_obj.py  #edit by myself
python cam_obj_tutorial.py	#copy code from tutorial
python object_detection_tutorial.py #dog code

================================= PART 3 ==================================================
========================= make own object detaction =======================================

#crop image >> .xml
Download image to image floder

https://github.com/tzutalin/labelImg
git clone https://github.com/tzutalin/labelImg.git
#Python 3 + Qt5
sudo apt-get install pyqt5-dev-tools
sudo pip3 install lxml
cd ~/labelImg
make qt5py3
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

================================= PART 4 =========================================================
========================= Creating TFRecords =====================================================
object-detection > images > test
                          > train 
                 > data
                 > training..object-detection.pbtxt
                           ..ssd_mobilenet_v1_pets.config
                 ..xml_to_csv.py
                 ..generate_tfrecord
                        
#install atom
https://codeforgeek.com/2014/09/install-atom-editor-ubuntu-14-04/
sudo add-apt-repository ppa:webupd8team/atom
sudo apt-get update
sudo apt-get install atom
***************************************************************
#xml_to_csv.py (convert to csv)
https://github.com/datitran/raccoon_dataset
cd object-detection
python xml_to_csv.py
python generate_tfrecord

#check 
~/Desktop/models/research
sudo python3 setup.py install

#generate tf record
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record


================================= PART 5 =========================================================
======================  train our object detection model =========================================


wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
extrack ssd_mobilenet_v1_coco_11_06_2017

***edit ssd_mobilenet_v1_pets.config change PATH_TO_BE_CONFIGURED and batch_size: 12 #24 if CPU error

Inside training dir, add object-detection.pbtxt:
item {
  id: 1
  name: 'macncheese'
}


#train
copy all floder from object-detection to /models/research/object_detection (mearg)

python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

#see view on TensorBoard
~/Desktop/models/research/object_detection
tensorboard --logdir='training'



================================= PART 6 ==============================================
============================== test our model =========================================


cd /object_detection

python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-10856 \
    --output_directory mac_n_cheese_inference_graph

****model.ckpt-xxxx see in training floder
**** if ERROR ****TypeError: x and y must have the same dtype, got tf.float32 != tf.int32
https://stackoverflow.com/questions/47242485/running-export-inference-graph-py-gives-difference-in-types-error

~/Desktop/models/research/object_detection/builders
You can find the post_processing_builder.py and change the function with 
    def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
  """Create a function to scale logits then apply a Tensorflow function."""
  def score_converter_fn(logits):
    cr = logit_scale
    cr = tf.constant([[cr]],tf.float32)
    print(logit_scale)
    print(logits)
    scaled_logits = tf.divide(logits, cr, name='scale_logits') #change logit_scale
    return tf_score_converter_fn(scaled_logits, name='convert_scores')
  score_converter_fn.__name__ = '%s_with_logit_scale' % (
      tf_score_converter_fn.__name__)
  return score_converter_fn

Then go to the research folder, run


python setup.py install
***************************************************


    
you should have a new directory, in my case, mine is mac_n_cheese_inference_graph, inside it, I have new checkpoint data, a saved_model directory, and, most importantly, the forzen_inference_graph.pb file.

past image you want to test in test_images floder

edit in object_detection_tutorial.ipynb (object_detection_custom.ipynb)
*****************************************
# What model to download.
MODEL_NAME = 'mac_n_cheese_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1

*************************************
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 8) ]
*************************************

-put images that you want to test in test_images floder and rename as image1.jpg,image2.jpg,image4.jpg
-edit in object_detection_custom.ipynb 
 TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 7) ]
 
#test
object_detection_custom.ipynb











