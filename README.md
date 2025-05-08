#  Single-Subject fMRI Task Classifier with Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify brain activity based on fMRI data from a single subject. Specifically, the model learns to detect whether the subject was viewing a **scissors** image at a given moment based on a single 2D axial brain slice extracted from a 4D fMRI volume.

---

##  Dataset

- **Source**: [Haxby 2001 Visual Object Recognition fMRI Dataset](https://openneuro.org/datasets/ds000105)
- **Subject**: `sub-1`
- **Modality**: 4D fMRI (`.nii.gz`) + task event labels (`.tsv`)
- **Resolution**: 40 × 64 × 64 × 121 (x, y, z, time)

---

##  Objective

Classify whether the subject was **viewing scissors** or **not** at a given timepoint using a single 2D slice of their fMRI brain scan.

---

##  Methods

- Parsed task event timings and aligned them with fMRI timepoints
- Extracted 2D axial brain slices at slice `z = 20`
- Normalized slices and labeled them as “scissors” or “not scissors”
- Built and trained a custom CNN using MATLAB's Deep Learning Toolbox
- Evaluated model performance using accuracy and confusion matrix

---

##  Results

- **Model**: Simple 2-layer CNN with ReLU + MaxPooling
- **Accuracy**: ~93% on the test set
- **Confusion Matrix**: Clean separation of “scissors” vs. “non-scissors” samples
- **Interpretation**: The CNN learned subject-specific activation patterns associated with visual perception of scissors


---

## 🧾 Citation

Haxby, J. V., et al. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science*, 293(5539), 2425–2430.
