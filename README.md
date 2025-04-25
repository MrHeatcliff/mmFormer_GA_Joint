# mmFormer_GA_Joint

This repository provides an implementation of the mmFormer architecture enhanced with Genetic Algorithm (GA) optimization for brain tumor segmentation using the BraTS 2018 dataset.

## 🧠 Overview

The project combines:
- **mmFormer**: a multi-modal transformer-based model for medical image segmentation.
- **Genetic Algorithm**: to optimize hyperparameters or configurations to improve segmentation performance.

---

## 📁 Project Structure

- `GA_segment.py`: Main script to run the Genetic Algorithm, run this to have the GA mask which is the input of out proposed system.
- `download2018.py`: Script to download the BraTS 2018 dataset.
- `preprocess.py`: Script for data preprocessing.
- `mmformer_GA_output/`: Stores results and model outputs.
- `requirements.txt`: List of Python dependencies (if missing, see below).

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MrHeatcliff/mmFormer_GA_Joint.git
cd mmFormer_GA_Joint
```
### 2. Install package

```bash
pip install requirement.txt
```
### 3. Download the dataset.
```bash
python download2018.py
```
### 4. Run preprocessing
- You should modidy the directory of dataset in preprocess.py, change to the absolute path in your computer.
```bash
python preprocess.py
```
### 5. Run GA
```bash
python GA_segment.py
```
### 6. Train the model
- After all the step, you should have 3 sub folder in dataset, 1 for input image, 1 for mask and 1 for ga mask, now you can train the model
```bash
cd mmFormer
./job.sh
```
### 7. Evaluation
```bash
python evaluation.py
```
## 📊 Results

### mmFormer vs. GA-Enhanced mmFormer

| Model        | Whole Tumor (%) | Tumor Core (%) | Enhancing Tumor (%) | Enhancing Tumor (Post-Process) (%) |
|--------------|------------------|----------------|----------------------|-------------------------------------|
| mmFormer     | **83.13**        | **72.02**      | **43.33**            | **44.94**                           |
| GA-mmFormer  | 79.54            | 69.60          | 39.95                | 42.75                               |

---

### Comparison with Other Models

#### Average Dice Coefficient

| Model       | Average DSC (%) |
|-------------|-----------------|
| U-Net       | 61.96           |
| GA-mmFormer | **66.16**       |

#### Per-Class Dice Comparison (GA-mmFormer vs. Residual U-Net)

| Model          | Whole Tumor (%) | Tumor Core (%) | Enhancing Tumor (%) |
|----------------|------------------|----------------|----------------------|
| Residual U-Net | 69.11            | 65.85          | **68.93**            |
| GA-mmFormer    | **79.54**        | **69.60**      | 39.95                |


---

## ✅ Conclusion

This project explores a hybrid approach combining Genetic Algorithms (GAs) with the mmFormer architecture for brain tumor segmentation on the BraTS 2018 dataset. While the proposed GA-mmFormer method introduced innovative preprocessing via GA-based mask generation, experimental results showed that it did not outperform the original mmFormer. However, it still demonstrated competitive performance, especially when compared to traditional models like U-Net and Residual U-Net.

The study highlights the complexity of integrating heuristic methods with deep learning, emphasizing the need for better fusion strategies and parameter tuning in future research.
