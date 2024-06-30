Team number: AOHW-307

Project name: Apple Defect Classification using Kria

Link to YouTube Video(s):
https://youtu.be/02gI8r25TnI

University name: Umons, Belgium / FSTF, Morocco

Participant(s):
- Nicolas Urbano Pintos (Co-director), Email: urbano.nicolas@gmail.com
- Alaeddine Ajouj, Email: allaeajouj9@gmail.com
- Elidrissi Errahhali Oumayma, Email: oumaymaelidrissi1@gmail.com

Supervisor name: Dr Carlos Valderrama
Supervisor e-mail: cavs63@gmail.com

Board used: Kria
Software Version:
- PyTorch version: 1.10.1
- Torchvision version: 0.11.2+cpu
- OpenCV version: 4.6.0
- Matplotlib version: 3.4.3
- NumPy version: 1.17.5
- Scikit-learn version: 0.24.2
- TQDM version: 4.64.0
- SciPy version: 1.3.1	
- h5py version: 2.10.0
- Pandas version: 1.3.1
- PIL version: 8.1.0

Brief description of project:
This project aims to develop an automated system for inspecting apples in both a plant collection pipeline and on trees. Using advanced image processing techniques and a camera, the system identifies various types of apples, assesses their quality, and determines the location, position, and distance of defective fruits.

Description of archive (explain directory structure, documents, and source files):

- `Distance/` - Contains files related to the distance estimation model using VGG16.
  - `VGG16_Distance.pth` - VGG16 model for distance estimation.
  - `evaluate_model.py` - Script to evaluate the distance model.
  - `train.py` - Script to train the distance model.
  - `VGG16_Distance.ipynb` - Jupyter notebook for distance model development.
  - `quantize.py` - Script to quantize the distance model.
  - `test_on_Kria.py` - Script to test the distance model on the Kria hardware.
  - `nyu_depth_v2_labeled.mat` - Dataset for training the distance model.
  - `evaluate_quantized_model.py` - Script to evaluate the quantized distance model.
  - `build/` - Directory containing build files for the distance model.
    - `quant_model/` - Contains quantized model files.
      - `quant_info.json` - Quantization information.
      - `Sequential_int.xmodel` - Quantized model.
      - `Sequential.py` - Model definition file.
      - `bias_corr.pth` - Bias correction file.
      - `__pycache__/` - Compiled Python files.
    - `compiled/` - Contains compiled model files.
      - `VGG16_Distance.xmodel` - Compiled distance model.
      - `md5sum.txt` - MD5 checksum file.
      - `meta.json` - Metadata for the compiled model.

- `Apple_Classification/` - Contains files related to the apple defect classification model.
  - `evaluate_model.py` - Script to evaluate the apple classification model.
  - `detection_classification.py` - Main script for apple defect detection and classification.
  - `quantize.py` - Script to quantize the apple classification model.
  - `train_model.py` - Script to train the apple classification model.
  - `VGG16_APPLE.ipynb` - Jupyter notebook for apple classification model development.
  - `detection_classification_utils.py` - Utility functions for apple classification and detection (YOLO + VGG16).
  - `evaluate_quantized_model.py` - Script to evaluate the quantized apple classification model.
  - `VGG16_APPLE.pth` - VGG16 model for apple classification.
  - `build/` - Directory containing build files for the apple classification model.
    - `quant_model/` - Contains quantized model files.
      - `quant_info.json` - Quantization information.
      - `Sequential_int.xmodel` - Quantized model.
      - `Sequential.py` - Model definition file.
      - `bias_corr.pth` - Bias correction file.
      - `Sequential_int_Apple.xmodel` - Additional quantized model file.
      - `__pycache__/` - Compiled Python files.
    - `compiled/` - Contains compiled model files.
      - `VGG16_APPLE.xmodel` - Compiled apple classification model.

  - `Test_model_on_Kria/` - Directory for testing the apple classification model on the Kria hardware.
    - `VGG16_APPLE.xmodel` - Compiled apple classification model for Kria.
    - `app.py` - Script to run the apple classification model on Kria.
    - `test_data/` - Directory containing Images used for testing on Kria.

  - `yolov3-coco/` - Directory containing YOLOv3 model files.
    - `coco-labels` - COCO dataset labels.
    - `get_model.sh` - Script to download the YOLOv3 model.
    - `yolov3.cfg` - YOLOv3 configuration file.
    - `yolov3.weights` - Pre-trained YOLOv3 weights.
    - `.ipynb_checkpoints/` - Jupyter notebook checkpoints.

  - `Apple/` - Directory containing apple datasets.
    - `Scab/` - Contains scab-affected apple datasets.
      - `Rotten_Green/` - Dataset of rotten green apples.
      - `Rotten_Red/` - Dataset of rotten red apples.
      - `Rotten_Yellow/` - Dataset of rotten yellow apples.
    - `healthy/` - Contains healthy apple datasets.
      - `Yellow/` - Dataset of yellow apples.
      - `Green/` - Dataset of green apples.
      - `Red/` - Dataset of red apples.
      - `.ipynb_checkpoints/` - Jupyter notebook checkpoints.

- `README.txt` - This readme file providing an overview of the project and archive contents.

- `Apple_Defect_Classification_Report.pdf` - Detailed report on the project methodology, results, and conclusions.

- `Apple_Defect_Classification_Video.mp4` - Video explaining the project overview, methodologies used, and demonstrations.

Instructions to build and test project:

1. **Apple Classification Model:**

   - Run cells in `Apple_Classification/VGG16_APPLE.ipynb` to build `VGG16_APPLE.xmodel`.
   - To test on Kria, transfer the `Test_model_on_Kria/` folder to your Petalinux Kria and run `app.py`.

2. **Apple Detection and Classification:**

   - To test detection and classification of apples on software, use `Apple_Classification/detection_classification.py` and `Apple_Classification/detection_classification_utils.py`.
   - Use the following commands:
     - To infer on an image stored on your local machine: `python3 yolo.py --image-path='/path/to/image/'`
     - To infer on a video stored on your local machine: `python3 yolo.py --video-path='/path/to/video/'`
     - To infer real-time on webcam: `python3 yolo.py`

3. **Distance Computing:**

   - Run cells in `Distance/VGG16_Distance.ipynb` to build `VGG16_Distance.xmodel`.
   - To test on Kria, use `Distance/test_on_Kria.py`.

For any questions or issues, contact the project participants or supervisor.

---

This version includes clearer instructions for building and testing each component of your project. Please adjust any details or add specific instructions as needed. If you have any further revisions or additions, feel free to let me know!

