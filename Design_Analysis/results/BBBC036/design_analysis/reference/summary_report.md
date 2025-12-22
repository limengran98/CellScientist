# CellScientist Design Analysis Report: BBBC036 Morphology Prediction

**Date:** 2025-12-22  
**Dataset:** BBBC036  
**Reference ID:** design_analysis_20251222_042028_Run3

---

## 1. Task Objectives and Problem Definition

**Objective:**  
To develop a robust computational pipeline for predicting drug-induced morphological changes in cells (Cell Painting data). The primary goal is to map chemical perturbations (SMILES + Dose) and the initial cellular state (Pre/Control) to the resulting phenotypic profile (Post/Treated).

**Problem Definition:**  
High-content screening data is plagued by batch effects and biological noise. The challenge is to disentangle technical variation (Plate effects) from biological signal (Compound effects) to allow for:
1.  **Phenotypic Profiling:** Predicting the morphological signature of a drug.
2.  **Mechanism of Action (MoA) Inference:** Grouping compounds by induced phenotype.
3.  **Toxicity Screening:** Identifying outliers in the morphological space.

---

## 2. Data Overview

**Data Source & Structure:**
- **Input Source:** `/data/users/limengran/CellScientist/Design_Analysis/data/BBBC036/CP_data.csv`
- **Processed Artifact:** `preprocessed_data.h5`

**Dimensionality & Statistics:**
- **Total Samples (Treated):** 416
- **Feature Space:** 50 selected morphological features (float64) after Variance Thresholding.
- **Control Samples (DMSO):** 84 (used for normalization).
- **Key Metadata:** `dose` (float), `smiles` (object), `plate_id` (object).

**Preprocessing Pipeline:**
1.  **Cleaning:** Handling of infinite values and NaN imputation.
2.  **Normalization:** Plate-wise robust normalization using **Median** and **MAD** of negative controls (DMSO). This centers the control distribution at 0 within each plate.
3.  **Pairing:** Construction of (Pre, Post) tuples, where 'Pre' is a randomly sampled DMSO vector from the *same plate* as the treated 'Post' sample.

---

## 3. Detailed Data Analysis Conclusions

### 3.1 Statistical Significance of Perturbations
Analysis of the primary variation components confirms that the treatment dosage is a statistically significant driver of morphological change.

*   **Linear Model (OLS) on PC1:**
    *   **Coefficient (Dose):** 0.1260
    *   **P-value:** 0.022 (< 0.05)
    *   **Conclusion:** There is a significant positive linear relationship between dose concentration and the first principal component of the morphological features. The model explains a portion of the variance, though the intercept (-0.6577) suggests a baseline shift.

### 3.2 Feature Redundancy & Correlation
*   **Observation:** Hierarchical clustering prompted warnings regarding "Large Matrix" operations, and PCA analysis indicated that a compressed latent representation is feasible.
*   **Conclusion:** The 50 selected features likely contain high multicollinearity (redundancy). Biological signals are distributed across correlated feature modules rather than independent channels.

### 3.3 Baseline Modeling Performance
*   **Model:** Simple Conditional Neural Network (Pre + Dose $\to$ Post).
*   **Training Dynamics (5 Epochs):**
    *   *Start:* Train Loss ~4.85, Val Loss ~4.70
    *   *End:* Train Loss ~4.62, Val Loss ~4.40
*   **Analysis:** The loss decreases monotonically but slowly, suggesting the simple dense architecture may be underfitting or the signal-to-noise ratio is low. The convergence of validation loss indicates no immediate overfitting.

---

## 4. Interpretability & Confounding Factors

### 4.1 Confounders
*   **Batch Effects (Plate ID):** Despite robust MAD normalization, plate-to-plate variation remains the most significant technical risk. If the model learns to predict 'Plate ID' rather than biology, generalization to new experiments will fail.
*   **Dose Distribution:** If doses are not uniformly distributed across compounds, the model might bias specific phenotypes to high-dose toxicity rather than specific MoAs.

### 4.2 Biological Interpretability
*   **Volcano Plots:** Identified distinct features separating high vs. low dose populations.
*   **Linearity Assumption:** The OLS analysis assumes a linear trajectory of phenotype evolution. Biological transitions (e.g., apoptosis, cell cycle arrest) are often non-linear/bifurcating, meaning linear projections may obscure critical transition states.

---

## 5. Reproducible Experiments & Validation Protocols

**Validation Strategy:**
- **Cross-Validation:** 5-Fold GroupKFold (split by `smiles` to prevent chemical leakage).
- **Metric:** Mean Squared Error (MSE) and Cosine Similarity between predicted and actual morphological vectors.

**Reproducibility Parameters:**
- **Random Seed:** `42`
- **Environment:** Python 3.11, PyTorch (CUDA), Scikit-Learn.
- **Hardware:** NVIDIA RTX 5880 Ada Generation.

**Baseline Experiment Results:**
| Epoch | Train Loss | Val Loss | Status |
|-------|------------|----------|--------|
| 1     | 4.8467     | 4.6985   | Converging |
| 5     | 4.6163     | 4.3974   | Stable |

---

## 6. Recommendations for Follow-Up Experiments

Based on the analysis and identified limitations, the following 8 experiments are proposed:

1.  **Multi-Modal Contrastive Learning (Chem-Bio CLIP)**
    *   *Action:* Train a dual-encoder (GNN for SMILES, MLP for Morphology) with InfoNCE loss.
    *   *Goal:* Learn a joint latent space where chemically similar drugs map to similar phenotypes.

2.  **Conditional Variational Autoencoder (cVAE)**
    *   *Action:* Implement a generative model $P(Post | Pre, SMILES, Dose)$ to generate full feature vectors.
    *   *Goal:* Capture the stochastic nature of cell responses and synthesize "virtual assays".

3.  **Adversarial De-biasing for Batch Correction**
    *   *Action:* Add a gradient reversal layer to the feature extractor that attempts to predict `Plate_ID`.
    *   *Goal:* Force the model to learn features that are invariant to plate batch effects.

4.  **Graph-Informed Morphological Attention**
    *   *Action:* Use an Attention mechanism to map molecular substructures (atoms/rings) to specific morphological feature clusters.
    *   *Goal:* Provide chemical interpretability (e.g., "This benzene ring drives nuclear enlargement").

5.  **Continuous Dose-Response Surface Modeling**
    *   *Action:* Replace scalar dose input with a monotonic constraint or Hypernetwork.
    *   *Goal:* Smoothly interpolate effects between tested doses to predict IC50 values.

6.  **Transformer-Based Sequence-to-Phenotype Translation**
    *   *Action:* Fine-tune ChemBERTa to "translate" SMILES tokens into morphological feature vectors.
    *   *Goal:* Leverage pre-trained chemical knowledge for rare scaffolds.

7.  **Multi-Task MoA Classification**
    *   *Action:* Add an auxiliary loss head to classify the Mechanism of Action while predicting regression targets.
    *   *Goal:* Regularize the latent space to align with known biological mechanisms.

8.  **Ablation Study: The "Pre" State Necessity**
    *   *Action:* Compare performance of $f(SMILES, Dose)$ vs. $f(SMILES, Dose, Pre_{Random})$.
    *   *Goal:* Quantify exactly how much information the "Control" state provides. If the delta is negligible, the pipeline can be simplified to ignore paired controls.

---

## 7. Risks, Biases, and Precautions

*   **Bias - Chemical Space:** The dataset splits by SMILES. If the chemical diversity is low (clustered scaffolds), the model will not generalize to novel chemical classes.
*   **Risk - Over-Smoothing:** Regression models (MSE loss) tend to predict the "average" phenotype, smoothing out extreme but biologically relevant outliers (toxicity signals). Generative models (cVAE) are recommended to mitigate this.
*   **Precaution - Leakage:** Ensure `GroupKFold` is strictly applied on SMILES. Random splitting will lead to massive leakage due to replicates/doses of the same compound appearing in both train and test sets.

---

## 8. Reproducibility Checklist

- [ ] **Input Data:** Ensure `CP_data.csv` is at `/data/users/limengran/CellScientist/Design_Analysis/data/BBBC036/`.
- [ ] **Random Seed:** Set `np.random.seed(42)` and `torch.manual_seed(42)`.
- [ ] **Device:** Check for CUDA availability (Code expects `cuda`, tested on RTX 5880).
- [ ] **Dependencies:** `fastcluster` is recommended for hierarchical clustering to avoid warnings.
- [ ] **Execution:** Run the normalization and splitting script before training. Check `preprocessed_data.h5` generation.
- [ ] **Output Path:** `/data/users/limengran/CellScientist/Design_Analysis/results/BBBC036/design_analysis/design_analysis_20251222_042028_Run3/`.