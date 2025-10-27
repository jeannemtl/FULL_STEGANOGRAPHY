# Install dependencies
pip install torch transformers datasets scipy matplotlib tqdm huggingface_hub

# Download the dataset
python data/download_esnli.py

# Fix transformers version conflict
pip uninstall transformers -y
pip install transformers==4.36.0

# Train the model
python training/train_esnli.py

# Test the trained model
python experiments/demo_esnli.py

# Evaluate bit recovery accuracy
python experiments/evaluate_esnli.py

### Expected Results
- Training loss: 0.3361 → 0.1586 (-52.8%)
- Bit recovery accuracy: 96.11%
- Training time: ~11.5 minutes
- Cost: ~$0.11 on RTX 4090
