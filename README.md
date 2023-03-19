# ONNXToTensorRT
C++ and Python convert ONNX to TensorRT

# quantization 
FP32\
FP16\
INT8

# Tested yolov5, yolov6, yolov7, yolov8 conversion successfully
yolov5: https://github.com/ultralytics/yolov5 \
yolov6: https://github.com/meituan/YOLOv6 \
yolov7: https://github.com/WongKinYiu/yolov7 \
yolov8: https://github.com/ultralytics/ultralytics

Note: If you use INT8 quantization, you need an additional file
C++：calibrator.cpp & calibrator.h
Python: calibrator.py

# Run
python main.py
