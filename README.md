# YOLO-WDD
YOLO-WDD is an improved YOLOv8 model based on an improved feature extractor and enhanced small-size object prediction.

**Requirements for setup**

This setup is designed for a Windows 10 environment with an NVIDIA RTX A2000 GPU.

1. Install Anaconda3 2024.06-1 (Python 3.12.4 64-bit)
2. Install Git-2.26.2-64-bit
3. Install your GPU driver
4. Check your compatible CUDA version and install CUDA. We have installed CUDA 12.8 version.
5. Choose CUDNN compatible with the CUDA version. For CUDA 12.8, we chose the CUDNN 8.9.7 version.
6. Create a new environment in Anaconda Navigator (Select Python 3.10 version)
7. Open the anaconda prompt, then activate the yolov8 environment by typing the below code and pressing enter
'''
   activate yolov8
'''
9. Navigate the directory of your computer, where the yolov8 root folder will be saved, we saved in E directory, so we type the below code and press enter

   E:
   
11. Then we created a folder that will be used as a yolov8 root folder. We created a folder name yolov8-gpu by types the below code and press enter

    mkdir yolov8-gpu
    
13. Then install PyTorch which can run on the GPU. We have install PyTorch 11.8 version. Copy the below link and paste it to the anaconda promt and press enter.

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  
14. Install yolov8, used the below command and press enter

    pip install untralytics

15. To train the model follow the below command, copy it, then paste, then press enter.

    yolo detect train model=yolov8n.yaml data=VSB_dataset.yaml imgsz=640 workers=8 batch=8 device=0 epochs=300 line_thickness=2

16. To test the model follow the below command

    yolo detect predict model=YOLO-WDD-best.pt source="VSB-dataset\images\test" save=True


