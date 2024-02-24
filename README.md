# Inkjet Printing Banding Defect Compensation

This repository contains the implementation of the banding defect compensation technique for inkjet printing, as described in the research paper "Research on Inkjet Printing Banding Defect Compensation Methods Based on Visual Perception Models" by Fang Xu, Lidu Wen, Yan Chen, and Hongwu Zhan from Zhejiang University of Technology, School of Mechanical Engineering, Hangzhou, China.

## Abstract

The research addresses the issue of banding in print products caused by nozzle failures in piezoelectric single-pass inkjet printers. A compensation technique based on a visual perception model is proposed, dynamically optimizing dot distribution and ink droplet size to mitigate banding defects. This technique utilizes a color difference compensation algorithm based on CIE2000 to adjust the dot structure at defect locations, significantly enhancing print quality and extending the lifespan of the print head.

## Keywords

- Piezoelectric Single-Pass Inkjet Printing
- Stripe Defect Compensation
- Visual Perception Model
- Halftone Image Optimization
- Chromatic Aberration Compensation (CIE2000)

## Installation

To run the compensation algorithm, ensure you have Python 3.6 or later installed. Clone this repository and install the required dependencies:

bash

git clone [https://github.com/your-repository/inkjet-printing-compensation.git](https://github.com/wenlidu123/Digital-Printing-Streak-Defect-Compensation)

cd Digital-Printing-Streak-Defect-Compensation

pip install -r requirements.txt
## Usage

To apply the banding defect compensation on an image, run the following command:
bash
python compensation_algorithm.py --image path/to/your/image.png
Replace `path/to/your/image.png` with the path to the image you wish to process.

## Experimental Results

The effectiveness of the proposed compensation technique has been validated through extensive experiments. Below are some of the experimental images showcasing the compensation results.

### Complex Color Structure Image

![Complex Color Structure Image](https://github.com/wenlidu123/Digital-Printing-Streak-Defect-Compensation/blob/main/Experimental%20images/Complex%20Color%20Structure%20Image.png?raw=true)

### Monotonous Color Structure Image

![Monotonous Color Structure Image](https://github.com/wenlidu123/Digital-Printing-Streak-Defect-Compensation/blob/main/Experimental%20images/Monotonous%20Color%20Structure%20Image.png?raw=true)

## Citation

If you find this work useful in your research, please consider citing:
Xu, F., Wen, L., Chen, Y., & Zhan, H. (Year). Research on Inkjet Printing Banding Defect Compensation Methods Based on Visual Perception Models. Zhejiang University of Technology, School of Mechanical Engineering, Hangzhou, China.
## Contact

For any queries regarding the implementation, please contact:

- Hongwu Zhan, Email: 201806040623@zjut.edu.cn

## License

This project is licensed under the MIT License - see the [LICENSE]([LICENSE](https://github.com/wenlidu123/Digital-Printing-Streak-Defect-Compensation/blob/main/LISENSE)https://github.com/wenlidu123/Digital-Printing-Streak-Defect-Compensation/blob/main/LISENSE) file for details.


