# Kronos Stock Predictor (demo)

## Setup (recommended)
1. Create a conda env (recommended for GPU PyTorch):
   - Example GPU (CUDA) install (use correct CUDA for your drivers):
     ```
     conda create -n kronos python=3.10 -y
     conda activate kronos
     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
     ```
   - Or install CPU-only torch via pip:
     ```
     pip install torch torchvision
     ```

2. Install Python deps:
pip install -r requirements.txt


3. (Optional but recommended) install Kronos tokenizer from GitHub:


pip install git+https://github.com/shiyu-coder/Kronos.git


4. Run the setup checker (interactive):


python setup_checker.py


5. Run Streamlit app:


streamlit run app.py


## Data
- The app uses `yfinance` to fetch live OHLCV history (no auth required).
- If you prefer offline data, download an AMZN CSV from Kaggle and use the "Use local CSV" option in the sidebar.

Kaggle examples (free, require Kaggle account):
- https://www.kaggle.com/datasets/varpit94/amazon-stock-data
- https://www.kaggle.com/datasets/adilshamim8/amazon-stock-price-history