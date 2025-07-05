import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from multimodal_dataset import MultimodalSpeechDataset
from multimodal_model import MultimodalSpeechModel
from utils import extract_vocab, text_to_indices, add_noise
from torchvision import transforms

CSV_PATH = 'MLIP/grid_metadata.csv'
BATCH_SIZE = 2
EPOCHS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Extract vocab
vocab_dict, inv_vocab_dict = extract_vocab(CSV_PATH)
num_classes = len(vocab_dict)

# 2. Dataset and DataLoader
video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def collate_fn(batch):
    audios, videos, transcriptions = zip(*batch)
    audios = [a.squeeze(0) if a.dim() == 2 and a.size(0) == 1 else a for a in audios]
    audios = [add_noise(a) for a in audios]
    audios = pad_sequence(audios, batch_first=True)  # [batch, time]
    videos = torch.stack(videos, dim=0)  # [batch, frames, C, H, W]
    targets = [torch.tensor(text_to_indices(t, vocab_dict), dtype=torch.long) for t in transcriptions]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return audios, videos, targets, target_lengths

dataset = MultimodalSpeechDataset(CSV_PATH, transform_video=video_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 3. Model, Loss, Optimizer
model = MultimodalSpeechModel(num_classes=num_classes).to(DEVICE)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for audios, videos, targets, target_lengths in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        audios = audios.to(DEVICE)
        videos = videos.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(audios, videos)  # [batch, seq_len, num_classes]
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [seq_len, batch, num_classes]
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(DEVICE)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}") 