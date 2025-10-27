# 0. Install dependencies
pip install torch transformers datasets scipy matplotlib tqdm huggingface_hub

# 1. Download the dataset
python data/download_esnli.py

# 2. Fix transformers version conflict
pip uninstall transformers -y
pip install transformers==4.36.0

# 3. Train the model
python training/train_esnli.py

# 4. Test the trained model
python experiments/demo_esnli.py

# 5. Evaluate bit recovery accuracy
python experiments/evaluate_esnli.py
