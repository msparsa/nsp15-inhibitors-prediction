# ChemBERTa and ML for NSP15 Inhibition Prediction

This repository provides a comprehensive workflow for predicting the inhibition of the NSP15 protein using machine learning models, ChemBERTa embeddings, and prompt engineering with ChatGPT.

## Repository Structure

- **`ml-checkpoints/`**  
  Contains pre-trained model checkpoints organized in two subfolders:
  - **`w_features/`**: Checkpoints for models trained on SMILES string embeddings with compound features.
  - **`wo_features/`**: Checkpoints for models trained on SMILES string embeddings without compound features.

- **`SMILES-DATA.csv`**  
  The dataset used in this project.  
  - Each row represents a compound described by its SMILES string.  
  - **Labels:** `0` indicates non-inhibiting, and `1` indicates inhibiting NSP15 protein.

- **`ChatGPT_classifier.ipynb`**  
  Notebook demonstrating prompt engineering for using ChatGPT as a classifier for the NSP15 inhibition task.

- **`ChemBERTa_and_ML.ipynb`**  
  Code for integrating ChemBERTa embeddings into machine learning models.

- **`Finetuning_ChemBERTa.ipynb`**  
  Code for fine-tuning the ChemBERTa model on the dataset.

- **`ML_models_playground.ipynb`**  
  A guide to using the pre-trained checkpoints stored in the `ml-checkpoints/` directory for prediction tasks.

- **`README.md`**  
  This file provides an overview of the repository.

---

## Key Features

1. **Dataset**  
   The dataset (`SMILES-DATA.csv`) contains SMILES strings annotated with binary labels:
   - `0`: Non-inhibiting compounds.
   - `1`: Inhibiting compounds.

2. **Model Checkpoints**  
   - Models trained with or without compound features are stored in `ml-checkpoints/`.
   - These checkpoints are ready for evaluation and further analysis using `ML_models_playground.ipynb`.

3. **ChemBERTa Fine-Tuning and Embedding**  
   - `Finetuning_ChemBERTa.ipynb` fine-tunes the ChemBERTa model for SMILES-based classification.
   - `ChemBERTa_and_ML.ipynb` showcases the use of ChemBERTa embeddings within custom machine learning models.

4. **ChatGPT Prompt Engineering**  
   - `ChatGPT_classifier.ipynb` explores prompt engineering techniques to leverage ChatGPT for classification tasks.

---

## How to Use

1. **Dataset**  
   - Use `SMILES-DATA.csv` for model training and evaluation.

2. **Fine-Tuning ChemBERTa**  
   - Run `Finetuning_ChemBERTa.ipynb` to fine-tune the ChemBERTa model on the dataset.

3. **Model Training and Embedding**  
   - Use `ChemBERTa_and_ML.ipynb` to generate embeddings and train models.

4. **Pre-Trained Checkpoints**  
   - Load pre-trained checkpoints from `ml-checkpoints/` using `ML_models_playground.ipynb`.

5. **Prompt Engineering with ChatGPT**  
   - Explore `ChatGPT_classifier.ipynb` to implement classification using ChatGPT.

---

## Requirements

- Python 3.8+
- Libraries:
  - PyTorch
  - Transformers (for ChemBERTa)
  - scikit-learn
  - pandas
  - Jupyter Notebook
- OpenAI API (for ChatGPT prompt engineering)

---

## Acknowledgments

This project leverages:
- **ChemBERTa**: For generating embeddings from SMILES strings.
- **ChatGPT**: For exploring natural language-based classification.
