# Vision Algorithms

## Feature Matching and Image Stitching

Report: [Link](https://docs.google.com/document/d/1kOSYU41aEM0BuxFiCCiIhXmkOMDkUoLPArne_ug2_J4/edit?usp=sharing)

## Requirements

```bash
pip3 install opencv-python
git clone https://github.com/aditya-jha13/vision-tasks
```

`opencv-python` is installed for implementing various vision algorithms required for the task.

## Installation

```bash
git clone https://github.com/aditya-jha13/vision-tasks
```

## Usage

```bash
python3 main1.py
```
`main1.py` uses `SIFT Feature Detector` and `BFMatcher Feature Matcher`
```bash
python3 main2.py
```
`main2.py` uses `ORB Feature Detector` and `BFMatcher Feature Matcher`
## Demo

### Input Images

Template Image

<img src="images/trainimage.jpeg" alt="train" width="400"/>

Transformed Image

<img src="images/queryimage.jpeg" alt="test" width="400"/>

### Detected Keypoints

Template Image

<img src="images/key1.png" alt="key1" width="400"/>

Transformed Image

<img src="images/key2.png" alt="key2" width="400"/>

### Stitched Image

<img src="images/final.png" alt="final" width="800"/>

## About

`SIFT detectors` or `ORB detectors` are used to detect `Keypoints` and their descriptions, furthermore Descriptor Matches are found using `Brute Force Matcher`. Then `.pt` attribute (Coordinates of Keypoints) are extracted from each Keypoints and a list of individual coordinates is created from matched descriptors for both Images. Finally `Homography Matrix` is found using those matched points and `warp perspective` is applied on the second Image to combine it with the First Image.

## Refrences

- Keypoints: [Link](https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html)
- Homography: [Link](https://docs.opencv.org/4.5.2/d9/dab/tutorial_homography.html)
- `cv::warpPerspective`: [Link](https://docs.opencv.org/4.5.2/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87)
