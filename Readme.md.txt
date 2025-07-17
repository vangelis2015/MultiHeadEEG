# 🧠 SSVEP EEG Pipeline

This repository provides a modular and extensible PyTorch pipeline for training, validating, and fine-tuning neural models on Steady-State Visual Evoked Potential (SSVEP) EEG data. The architecture is designed for reproducible research, supporting multi-head Transformer-based models, per-subject fine-tuning, and advanced loss functions such as Deep CCA and reconstruction loss.

---

##  Features

-  Two-stage training (global pretraining + subject-specific fine-tuning)
-  Multi-head Transformer encoder for temporal and spatial EEG representation
-  Integration of auxiliary objectives (reconstruction, Deep CCA)
-  Block-based evaluation (e.g., Leave-One-Block-Out)
-  Modular design for training/validation loops and model customization
-  Easy configuration via `config.py`
-  Compatibility with MLflow and experiment tracking tools

---

## Project Structure

MultiHeadEEG/
├── main.py # Entry point for training
├── train.py # Main training (and validation) routine
├── config.py # All configuration parameters
├── modelST_CLSToken.py # Definition of the EEG Transformer model
├── HelpWithMatlab.py # Utilities for filtering, normalization, alignment
├── load_utils.py # Subject loading, data splitting, etc
├── requirements.txt
└── README.md



---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/ssvep-pipeline.git
cd ssvep-pipeline
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt


## Run training:

	cd ssvep_pipeline
	python main.py

## Configuration:

All parameters can be modified in config.py, including:

	seg_time = 50
	batch_size = 16
	subject_ids = range(1, 36)
	folder = "C:/path/to/your/benchmark_dataset"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Dependencies

Key libraries:

    torch

    numpy

    scikit-learn

    tqdm

    mlflow (optional)


Install them using:

	pip install -r requirements.txt

## License

This project is licensed under the MIT License.

##Contact

Developed by [Your Name]
📧 your.email@example.com
🔗 github.com/your-username

