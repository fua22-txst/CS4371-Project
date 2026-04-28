# CS4371 Project README
by Cade, Miguel, Nikolas, Lucian


The README file in the GitHub repository indicates clearly how to easily clone and build/deploy the code. The README file in the GitHub repository clearly indicates what functionality does (and does not, if applicable) work in the final version of the application. Additionally, it should include references to two scholarly papers: one representing prior research, serving as the foundational bedrock for the current study, and another representing contemporary work that acknowledges and builds upon the findings of the current paper. This practice is akin to the methodology of an archeologist, meticulously documenting the lineage of ideas and advancements within the academic landscape.

__Overview__
This project uses four different models to detect network intrusions. The goal was to fiddle with the idea of improving overall detection by having a different model to cover the weaknesses of the CNN. 

__Functionality__:
Along with the original CNN, our code trains 3 additional ML models on the entire provided dataset to test if we can improve the results from the original CNN. We also did an additional set of training to cover up the weaknesses of the original model, training 3 models on Benign and Spoofing attacks only to see if we could get better results in these areas. Our code also prints the results and metrics of the models to compare to the original. We also had to add several extra files upon merging all our our code together for it to function properly. But to run and see the results of our code, all you need to worry about is main.py.



__How to Get Started With Our Code__

1. Clone our GitHub repository onto your machine, using our link: https://github.com/fua22-txst/CS4371-Project 
2. Create and activate a virtual environment:
  a. Python -m venv venv
  b. Windows- venv\Scripts\activate. Mac- source venv/bin/activate
  c. Colab
3. Install Dependencies: pip install -r requirements.txt
4. Download the dataset (https://txst-my.sharepoint.com/:u:/g/personal/fua22_txstate_edu/IQDizzZi_Ja0TJS4jCuf0pxaAd7t2m7IMHPT3AX55RZUEDA?e=hcYpuS) and put the files in the correct places:
  a. Extract and place the CSV files in the appropriate directories (‘data/train/’ and ‘data/test/’) 
  b. Within the ‘data/’ directory, there should be ‘train/’ and ‘test/’ folders to hold the CSV files for training and testing. Move the files from the data set to each of these
5. Make sure you are in the src directory, then run “python main.py”, and our code should execute


Note: The dataset SAMPLE_SIZE is hardcoded to 100 items from the dataset for the sake of the demo (75 items for the second CNN/Gradient Boost/Random Forest). This value is located in `*_data_loader.py` for the corresponding model. To run the models on larger training datasets, change this value for each model's `*_data_loader.py`. For the second layer CNN, comment out line 111 in `data_loader_spoofing.py` to run it on the full dataset.


Prior research paper(cited in our paper): https://ieeexplore.ieee.org/document/11455983
  This paper is referenced by ours and was influential because it showed how to use machine learning models trained on IoMT data to perform binary and multi level attack classification. Our paper used a CNN trained on IoMT data following some of the ideas presented in this paper. 


Post paper research(cites our paper): https://www.sciencedirect.com/science/article/pii/S2542660525001453
  This paper references our paper, and is basically doing a study comparing the effectiveness of ML models for cybersecurity in healthcare, when they are trained on IoT datasets vs when they are trained on IoMT datasets. They use the same IoMT dataset that the researchers in our paper used, and reference the results    of our papers researchers.

-------------------------------------------------------------------------------------------------------------------------------------------

# ORIGINAL PAPER'S README (kept for reference)
# Securing Healthcare with Deep Learning: A CNN-Based Model for medical IoT Threat Detection

<div align="center">

[![IEEE Conference](https://img.shields.io/badge/IEEE-ICIS%202024-blue.svg)](https://doi.org/10.1109/ICIS64839.2024.10887510)
[![arXiv](https://img.shields.io/badge/arXiv-2410.23306-b31b1b.svg)](https://arxiv.org/abs/2410.23306)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Presentation](https://img.shields.io/badge/YouTube-Presentation-red.svg)](https://youtu.be/hPV5H9kTbYM?si=fWtb_eaIiLQ3uGEy)

*Official Implementation - Presented at the 19th Iranian Conference on Intelligent Systems (ICIS 2024)* **IEEE Indexed**
</div>


---

## Overview
This repository provides the implementation of our CNN-based intrusion detection model for Internet of Medical Things (IoMT) systems. The model performs multi-class classification on network traffic, distinguishing 19 attack types as well as benign traffic using the CICIoMT2024 dataset. All source code corresponds to the work presented in our paper. If you use or reference this repository, please **cite** [our paper](#-citation). This work has received **15+ citations** and **20+ GitHub stars**.

**Key Features:**
- Multi-classification support: Binary (2-class), Categorical (6-class), and Multiclass (19-class)
- Perfect accuracy of 0.99 across all classification tasks
- Outperforms previous state-of-the-art methods
---

## Performance Metrics

<div align="center">

![Model Comparison](https://github.com/user-attachments/assets/7dc2bd46-c2ea-49cb-b94f-7ee42b268d56)

*Performance comparison across different classification tasks*

</div>

---

## Model Architecture

<div align="center">

![CNN Architecture](https://github.com/user-attachments/assets/e76e8cb4-a185-4726-abcb-b50482786088)

*Architecture of the CNN model for IoMT attack classification*

</div>

---

## 🚀 Quick Start

### Step 1: Clone Repository
```bash
git clone https://github.com/alirezamohamadiam/Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection.git
```
> Make sure Git is installed on your machine. If not, grab it from: https://git-scm.com/install

### Step 2: Install Requirements
Navigate to Project Directory
```bash
cd Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection
```
then:
```bash
pip install -r requirements.txt
```
> **Note:** Python 3.7+ is required

### Step 3: Download Dataset
Download the **CIC IoMT Dataset 2024** from:  
🔗 [https://www.unb.ca/cic/datasets/iomt-dataset-2024.html](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)

### Step 4: Prepare Data
Place the CSV files in the data directories:
```
data/
├── train/     ← Put training CSV files here
└── test/      ← Put testing CSV files here
```
> **Detailed instructions:** See [`README_DATA.md`](https://github.com/alirezamohamadiam/Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection/blob/main/README_DATA.md)

### Step 5: Run Training
Navigate to the `src` directory:
```bash
cd src
```
To run the model, execute `main.py` and specify the classification configuration:
```bash
python main.py --class_config <num_classes>
```

Replace `<num_classes>` with:
- **2** for binary classification,
- **6** for categorical,
- **19** for multiclass.

**Example (binary classification):**
```bash
python main.py --class_config 2
```
---

## 📂 Project Structure

```
project/
├── data/
│   ├── train/            # Training CSV files (see README_DATA.md)
│   └── test/             # Testing CSV files (see README_DATA.md)
├── src/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── model.py          # CNN model definition and training
│   └── main.py           # Main execution script
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation (this file)
└── README_DATA.md        # Data preparation guide
```
---

## 📚 Citation

If you use this code or find our work helpful, please cite our paper:
```bibtex
@inproceedings{mohammadi2024securing,
  title={Securing Healthcare with Deep Learning: A CNN-Based Model for medical IoT Threat Detection},
  author={Mohammadi, Alireza and Ghahramani, Hosna and Asghari, Seyyed Amir and Aminian, Mehdi},
  booktitle={2024 19th Iranian Conference on Intelligent Systems (ICIS)},
  pages={168--173},
  year={2024},
  organization={IEEE}
}
```
**Plain Text Citation (APA style):**
```
Mohammadi, A., Ghahramani, H., Asghari, S. A., & Aminian, M. (2024, October). Securing Healthcare with Deep Learning: A CNN-Based Model for medical IoT Threat Detection. In 2024 19th Iranian Conference on Intelligent Systems (ICIS) (pp. 168-173). IEEE.
```

**Links:**
The open-access PDF version of the paper is available on arXiv. You can read the full article there for free. 

* 📄 **IEEE Xplore (official version, not open access):**
  [https://doi.org/10.1109/ICIS64839.2024.10887510](https://doi.org/10.1109/ICIS64839.2024.10887510)

* 📄 **arXiv (PDF and open-access version):**
  [https://arxiv.org/abs/2410.23306](https://arxiv.org/abs/2410.23306)
---

<div align="center">

**⭐ If you find this work useful, please consider giving it a star! ⭐**

Made with ❤️ by Alireza Mohamadi

</div>
































