import torch

MODEL_SAVE_PATH = "models/saved_models/cnn.pt"
LOAD_SAVED_MODEL = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_SAVE_PATH).to(device)