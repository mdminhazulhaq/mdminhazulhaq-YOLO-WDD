# YOLO-WDD
YOLO-WDD is an improved YOLOv8 model based on an improved feature extractor and enhanced small-size object prediction.

**Environmental Setup prerequisites**

This setup is designed for a Windows 10 environment with an NVIDIA RTX A2000 GPU.

1. Install Anaconda3 2024.06-1 (Python 3.12.4 64-bit)
2. Install Git-2.26.2-64-bit
3. Install your GPU driver
4. Check your compatible CUDA version and install CUDA. We have installed CUDA 12.8 version.
5. Choose CUDNN compatible with the CUDA version. For CUDA 12.8, we chose the CUDNN 8.9.7 version.

7. Create a new environment in Anaconda Navigator (Select Python 3.10 version), we create an environment called yolov8
8. Open the anaconda prompt, then activate the yolov8 environment by typing the below code and pressing enter
```
   activate yolov8
```
9. Navigate the directory of your computer, where the yolov8 root folder will be saved, we saved in the E directory, so we type the below code and press enter
```
   E:
```
11. Then we created a folder that will be used as a yolov8 root folder. We created a folder named yolov8-gpu by typing the below code and pressing enter
```
    mkdir yolov8-gpu
```
13. Then install PyTorch which can run on the GPU. We have installed PyTorch 11.8 version. Please copy the link below, paste it to the anaconda promt, and press enter.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
14. Installed ultralytics, use the below command and press enter
```
    pip install untralytics
```

**Data preparation**
1.	Locate the image and label file of respective datasets in the ‘YOLO-WDD-Dataset’ folder.
2.	Create a directory file for respective datasets that contains the images and their label text files.
3.	Create .data files for both datasets and locate in the ‘cfg’ folder. The file contains the following information:
```
# Dataset Path
path: E:\yolov8-gpu\dataset\wood-defect-dataset

train: images\train # relative to path
val: images\val # relative to path
test: images\test # relative to path

# Class Names
names: ['live knot', 'dead knot', 'knot with crack', 'crack', 'resin', 'knot missing', 'marrow']
```
4. Locate the YOLO-WDD.cfg file in the ‘cfg’ folder.


**Training and test of the YOLO-WDD model and YOLOv8n model**
1. Training command for YOLO-WDD model
```
yolo detect train model=YOLO-WDD.yaml data=dataset\wood-defect.yaml imgsz=640 workers=8 batch=8 device=0 epochs=300 line_thickness=2 patience=300
```
2. Test command for YOLO-WDD model
```
yolo detect predict model=YOLO-WDD-best.pt source="E:\yolov8-gpu\dataset\wood-defect-dataset\images\test" save=True
```
3. Training command for YOLOv8n model
```
yolo detect train model=yolov8n.yaml data=dataset\wood-defect.yaml imgsz=640 workers=8 batch=8 device=0 epochs=300 line_thickness=2 patience=300
```
4. Test command for YOLOv8n model
```
yolo detect predict model=YOLOv8n-best.pt source="E:\yolov8-gpu\dataset\wood-defect-dataset\images\test" save=True
```


**Training and test of the YOLOv10n model**
1. Training command for YOLOv10n model
```
yolo detect train model=yolov10n.yaml data=dataset\wood-defect.yaml imgsz=640 workers=8 batch=8 device=0 epochs=300 patience=300 project=results name=yolov10n-batch8-epochs300 amp=True
```
2. Test command for YOLOv10n model
```
yolo detect predict model=YOLOv10n-best.pt source=assets
```

**Training and test of the YOLOv7-tiny model**
1. Training command for YOLOv7-tiny model
```
python train.py --img 640 --batch 8 --epochs 300 --data dataset/wood-defect.yaml --cfg yolov7-tiny.yaml --weights yolov7-tiny.py --device 0 --workers 8
```
2. Test command for YOLOv7-tiny model
```
python detect.py --weights YOLOv7-tiny-best.pt source E:\yolov8-gpu\dataset\wood-defect-dataset\images\test
```


**Jetson Nano Setup**
This setup is for the NVIDIA Jetson Nano, which features a Maxwell GPU, Quad-Core ARM Cortex-A57 Processor, and 4GB LPDDR4 Memory.

1. Flash the microSD card with JetPack 4.4 – Download and write the JetPack 4.4 image to a microSD card using balenaEtcher or a similar tool.
2. Boot and install the OS on Jetson Nano – Insert the flashed microSD card into the Jetson Nano, connect peripherals (keyboard, mouse, display, and power supply), and follow the on-screen setup instructions to complete the installation.

Now, open the Jetson Nano terminal and follow the steps one by one to configure it for testing YOLO model performance.
