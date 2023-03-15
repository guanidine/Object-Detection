import torch
import torch.optim as optim
from tqdm import tqdm

import config
from loss import YoloLoss
from model import YOLOv1
from utils import (
    mean_avg_precision,
    get_evaluation_bboxes,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    plot_couple_examples
)


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if config.LOAD_MODEL:
        load_checkpoint(torch.load(config.LOAD_MODEL_FILE), model, optimizer, config.LEARNING_RATE)

    train_loader, test_loader = get_loaders(
        train_csv_path=config.DATASET + "/100examples.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    for epoch in range(config.NUM_EPOCHS):
        if config.TEST_MODE:
            plot_couple_examples(model, test_loader)
            import sys
            sys.exit()

        pred_boxes, target_boxes = get_evaluation_bboxes(
            train_loader, model, iou_threshold=0.5, prob_threshold=0.4
        )

        mean_avg_prec = mean_avg_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        if config.SAVE_MODEL and mean_avg_prec > 0.9:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
