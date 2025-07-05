import torch
from torch.utils.data import DataLoader
from multimodal_dataset import MultimodalSpeechDataset
from multimodal_model import MultimodalSpeechModel
from utils import extract_vocab, indices_to_text, text_to_indices
from torchvision import transforms
import torch.nn.functional as F

CSV_PATH = 'grid_metadata.csv'
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load vocab
vocab_dict, inv_vocab_dict = extract_vocab(CSV_PATH)
num_classes = len(vocab_dict)

# Video transform
video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = MultimodalSpeechDataset(CSV_PATH, transform_video=video_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = MultimodalSpeechModel(num_classes=num_classes).to(DEVICE)
model.eval()
# Optionally: model.load_state_dict(torch.load('best_model.pth'))

def greedy_decode(log_probs, inv_vocab_dict):
    pred_indices = log_probs.argmax(-1).cpu().numpy()
    # Collapse repeats and remove blanks (0)
    prev = -1
    pred_str = ''
    for idx in pred_indices[0]:
        if idx != prev and idx != 0:
            pred_str += inv_vocab_dict.get(idx, '')
        prev = idx
    return pred_str

with torch.no_grad():
    for i, (audio, video, gt_text) in enumerate(dataloader):
        audio = audio.to(DEVICE)
        video = video.to(DEVICE)
        logits = model(audio, video)  # [batch, seq_len, num_classes]
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [seq_len, batch, num_classes]
        pred_text = greedy_decode(log_probs, inv_vocab_dict)
        print(f"Sample {i}:\n  GT:  {gt_text[0]}\n  Pred: {pred_text}\n") 