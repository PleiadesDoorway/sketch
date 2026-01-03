import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.seq_transfomer import SketchTransformer
from dataset import Quickdraw414k4VanillaTransformer

# =========================
# Config
# =========================
MAX_LEN = 100
BATCH_SIZE = 64
NUM_EPOCHS = 50
NUM_CLASSES = 345
LR = 5e-4

TRAIN_SKETCH_LIST = "../dataset/QuickDraw414k/picture_files/tiny_train_set.txt"
TEST_SKETCH_LIST  = "../dataset/QuickDraw414k/picture_files/tiny_test_set.txt"

TRAIN_ROOT = "../dataset/QuickDraw414k/coordinate_files/train"
TEST_ROOT  = "../dataset/QuickDraw414k/coordinate_files/test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Sketch preprocessing
# =========================
def process_one_sketch(npy_path):
    raw = np.load(npy_path, allow_pickle=True, encoding="latin1")

    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        data = raw
    else:
        data = raw.item() if raw.shape == () else raw[0]

    coord = data[:, :2].astype(np.float32)
    pen_down = data[:, 2].astype(np.float32)

    valid = (data[:, 2] + data[:, 3]) > 0
    stroke_len = int(valid.sum())

    return coord, pen_down, stroke_len


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for (
        coordinate,
        label,
        flag_bits,
        stroke_len,
        attention_mask,
        padding_mask,
        position_encoding
    ) in dataloader:

        coordinate = coordinate.to(DEVICE).float()
        flag_bits = flag_bits.to(DEVICE).float()
        attention_mask = attention_mask.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(
            coordinate,
            flag_bits,
            padding_mask,
            attention_mask
        )

        loss = criterion(logits, label)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    return total_loss / len(dataloader), correct / total


# =========================
# Main
# =========================
def main():
    # =========================
    # Build TRAIN data_dict
    # =========================
    print("Building train data_dict...")
    train_data_dict = {}

    with open(TRAIN_SKETCH_LIST) as f:
        for line in f:
            png_path, label = line.strip().split()
            npy_rel_path = png_path.replace("png", "npy")
            npy_path = os.path.join(TRAIN_ROOT, npy_rel_path)

            if not os.path.exists(npy_path):
                raise FileNotFoundError(npy_path)

            coord, flag_bits, stroke_len = process_one_sketch(npy_path)

            train_data_dict[npy_rel_path] = (
                torch.from_numpy(coord),
                torch.from_numpy(flag_bits),
                stroke_len
            )

            if len(train_data_dict) % 1000 == 0:
                print(f"Train sketches: {len(train_data_dict)}")

    print(f"Train sketches: {len(train_data_dict)}")

    # =========================
    # Build TEST data_dict
    # =========================
    print("Building test data_dict...")
    test_data_dict = {}

    with open(TEST_SKETCH_LIST) as f:
        for line in f:
            png_path, label = line.strip().split()
            npy_rel_path = png_path.replace("png", "npy")
            npy_path = os.path.join(TEST_ROOT, npy_rel_path)

            if not os.path.exists(npy_path):
                raise FileNotFoundError(npy_path)

            coord, flag_bits, stroke_len = process_one_sketch(npy_path)

            test_data_dict[npy_rel_path] = (
                torch.from_numpy(coord),
                torch.from_numpy(flag_bits),
                stroke_len
            )

            if len(test_data_dict) % 1000 == 0:
                print(f"Test sketches: {len(test_data_dict)}")

    print(f"Test sketches: {len(test_data_dict)}")

    # =========================
    # Dataset & DataLoader
    # =========================
    train_dataset = Quickdraw414k4VanillaTransformer(
        sketch_list=TRAIN_SKETCH_LIST,
        data_dict=train_data_dict
    )

    test_dataset = Quickdraw414k4VanillaTransformer(
        sketch_list=TEST_SKETCH_LIST,
        data_dict=test_data_dict
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  
        pin_memory=False
    )



    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # Model
    # =========================
    model = SketchTransformer(
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    # =========================
    # Training loop
    # =========================
    best_test_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for (
            coordinate,
            label,
            flag_bits,
            stroke_len,
            attention_mask,
            padding_mask,
            position_encoding
        ) in pbar:

            coordinate = coordinate.to(DEVICE).float()
            flag_bits = flag_bits.to(DEVICE).float()
            attention_mask = attention_mask.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            logits = model(
                coordinate,
                flag_bits,
                padding_mask,
                attention_mask
            )

            loss = criterion(logits, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            pbar.set_postfix(
                loss=total_loss / (total + 1e-6),
                acc=correct / total
            )

        scheduler.step()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        test_loss, test_acc = evaluate(model, test_loader, criterion)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
        )

    print("Training finished.")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    main()
