## MSDS 462-DL Final Project
### People Analytics for Radiation Treatment Rooms

![Inference examples](https://github.com/anrobr/MSDS_462-DL-Final_Project/blob/main/header_image.png?raw=true)

---
### Project Goal
Automate the counting of people in radiation treatment rooms to avoid accidental irradiation of medical staff.

---
### Approach
Use real-time object tracking based on 

- YOLO v4 tiny
- MobileNetV2-SDDLite
- MobileNetV2-OpenPose

to track the number of entries/exits to/from a radiation treatment room on the Intel Neural Compute Stick 2 (NCS2).

---
### Tags
[Computer vision, object detection, object tracking, deep-learning, real-time, edge AI, OpenVINO, Intel Neural Compute Stick 2, NCS2.]

---
### Repository Contents

* documentation
  * Synopsis.pdf
  * Management Presentation Deck.pptx

* implementation
  * converted_models (_models in the OpenVINO intermediate representation_)
  * labels (_class labels for the YOLO v4 tiny and the MobileNetV2-SDDLite detectors_)
  * raw_models (_TensorFlow models used for the conversion to the OpenVINO intermediate representation_)
  * utils (_utility implementation for computations, visualizations, and so forth_) 
  * **maze-entry-tracking.ipynb** (_the Jupyter notebook containing the final project implementation_)
