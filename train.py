"""Train, validate, and test the ASDGC forecasting model."""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from net import ASDGC
from util import metric, normalize_data


DATA_ROOT = "./dataset"
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2


def set_seed(seed):
    """Set random seeds for reproducible model runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info("Random seed set to %d", seed)


def load_dataset(data_name, seq_len, pred_len, norm_type="global", train_ratio=TRAIN_RATIO):
    """Load a comma-separated dataset and create forecasting windows."""
    start_time = time.time()
    data_path = os.path.join(DATA_ROOT, data_name, f"{data_name}.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    try:
        data = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
    except Exception:
        logging.exception("Failed to read dataset: %s", data_path)
        raise

    if data.ndim == 1:
        data = data[:, np.newaxis]

    total_steps, num_variables = data.shape
    num_samples = total_steps - seq_len - pred_len + 1
    if num_samples <= 0:
        raise ValueError(
            "The dataset is too short for the requested sequence and prediction lengths."
        )

    train_samples = int(train_ratio * num_samples)
    fit_end_idx = train_samples + seq_len + pred_len - 1

    if norm_type == "revin":
        mean = np.zeros(num_variables, dtype=np.float32)
        std = np.ones(num_variables, dtype=np.float32)
        logging.info("Normalization: RevIN inside the model; global normalization skipped.")
    elif norm_type in {"global", "global_revin"}:
        data, mean, std = normalize_data(data, fit_end_idx=fit_end_idx)
        logging.info(
            "Normalization: %s using training-range statistics from raw steps [0, %d).",
            norm_type,
            fit_end_idx,
        )
    else:
        raise ValueError(
            "Unsupported norm_type. Use 'global', 'revin', or 'global_revin'."
        )

    inputs = []
    targets = []
    for start in range(num_samples):
        inputs.append(data[start : start + seq_len].T)
        targets.append(data[start + seq_len : start + seq_len + pred_len].T)

    x = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    logging.info(
        "Dataset loaded: steps=%d, variables=%d, samples=%d, input_shape=%s, target_shape=%s, elapsed=%.2fs",
        total_steps,
        num_variables,
        len(x),
        tuple(x.shape),
        tuple(y.shape),
        time.time() - start_time,
    )
    return x, y, mean, std


def save_results(save_dir, results, data_name, pred_len, timestamp):
    """Save model-run and test metrics as JSON and append a compact summary."""
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(
        save_dir, f"{data_name}_predlen_{pred_len}_{timestamp}.json"
    )
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    summary_path = os.path.join(save_dir, "results_summary.txt")
    with open(summary_path, "a", encoding="utf-8") as file:
        file.write(
            f"[{timestamp}] dataset={data_name}, norm_type={results['norm_type']}, "
            f"seq_len={results['seq_len']}, pred_len={pred_len}, "
            f"best_val_loss={results['best_val_loss']:.6f}, "
            f"test_RSE={results['test_metrics']['RSE']:.6f}, "
            f"test_RAE={results['test_metrics']['RAE']:.6f}, "
            f"test_CORR={results['test_metrics']['CORR']:.6f}\n"
        )
    return result_path


def unwrap_model(model):
    """Return the underlying model when DataParallel is active."""
    return model.module if isinstance(model, nn.DataParallel) else model


def load_model_state(model, state_dict):
    """Load checkpoints saved with or without a DataParallel prefix."""
    cleaned_state = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        logging.warning("Missing checkpoint parameters: %s", sorted(missing))
    if unexpected:
        logging.warning("Unexpected checkpoint parameters: %s", sorted(unexpected))


def train_model(args, timestamp):
    """Run training, validation, checkpointing, and final testing."""
    set_seed(args.seed)
    run_start = time.time()

    device = torch.device("cpu")
    device_ids = []
    if torch.cuda.is_available() and args.gpus:
        device_ids = args.gpus
        device = torch.device(f"cuda:{device_ids[0]}")
        logging.info("Using GPU devices %s; primary device is %s", device_ids, device)
    else:
        logging.info("Using CPU")

    logging.info("Run arguments:\n%s", json.dumps(vars(args), indent=2))

    x, y, mean, std = load_dataset(
        args.data,
        args.seq_len,
        args.pred_len,
        norm_type=args.norm_type,
        train_ratio=TRAIN_RATIO,
    )

    total_size = len(x)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Train, validation, and test splits must all contain samples.")

    logging.info(
        "Dataset split: train=%d, validation=%d, test=%d",
        train_size,
        val_size,
        test_size,
    )

    train_dataset = TensorDataset(x[:train_size], y[:train_size])
    val_dataset = TensorDataset(
        x[train_size : train_size + val_size],
        y[train_size : train_size + val_size],
    )
    test_dataset = TensorDataset(
        x[train_size + val_size :],
        y[train_size + val_size :],
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    base_model = ASDGC(
        num_variables=x.shape[1],
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_scales=args.num_scales,
        dropout=args.dropout,
        norm_type=args.norm_type,
    )

    checkpoint = None
    start_epoch = 0
    best_val_loss = float("inf")
    if args.pretrained_model:
        if not os.path.exists(args.pretrained_model):
            logging.warning(
                "Pretrained model not found: %s. Training from scratch.",
                args.pretrained_model,
            )
        else:
            try:
                checkpoint = torch.load(
                    args.pretrained_model,
                    map_location="cpu",
                    weights_only=True,
                )
                state_dict = (
                    checkpoint["model_state_dict"]
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                    else checkpoint
                )
                load_model_state(base_model, state_dict)
                logging.info("Loaded pretrained model: %s", args.pretrained_model)
            except Exception:
                checkpoint = None
                logging.exception(
                    "Failed to load pretrained model. Training from scratch."
                )

    base_model = base_model.to(device)
    model = (
        nn.DataParallel(base_model, device_ids=device_ids)
        if len(device_ids) > 1
        else base_model
    )
    if len(device_ids) > 1:
        logging.info("DataParallel enabled on devices %s", device_ids)

    criterion = nn.HuberLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    if args.resume_training and isinstance(checkpoint, dict):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
            logging.info(
                "Resumed checkpoint at epoch %d with best validation loss %.6f",
                start_epoch,
                best_val_loss,
            )
        except KeyError:
            logging.warning(
                "Checkpoint does not contain complete optimizer and scheduler state."
            )

    save_dir = "save"
    best_model_dir = os.path.join(save_dir, "best_model")
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    early_stopping_counter = 0
    best_epoch = start_epoch
    best_val_metrics = {}
    model_path = os.path.join(
        best_model_dir,
        (
            f"ASDGC_{args.data}_normType_{args.norm_type}_seqLen_{args.seq_len}_"
            f"predLen_{args.pred_len}_numScales_{args.num_scales}_"
            f"batchSize_{args.batch_size}_epochs_{args.epochs}_seed_{args.seed}_"
            f"dropout_{args.dropout}_{timestamp}.pth"
        ),
    )

    logging.info("Training started")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0

        for batch_index, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=pin_memory)
            batch_y = batch_y.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y.permute(0, 2, 1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * batch_x.size(0)

            if batch_index % 10 == 0:
                logging.debug(
                    "Epoch %d, batch %d/%d, loss %.6f",
                    epoch + 1,
                    batch_index,
                    len(train_loader),
                    loss.item(),
                )

        model.eval()
        val_loss_sum = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=pin_memory)
                batch_y = batch_y.to(device, non_blocking=pin_memory)
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y.permute(0, 2, 1))
                val_loss_sum += loss.item() * batch_x.size(0)
                val_predictions.append(prediction.cpu().numpy())
                val_targets.append(batch_y.permute(0, 2, 1).cpu().numpy())

        avg_train_loss = train_loss_sum / train_size
        avg_val_loss = val_loss_sum / val_size

        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_predictions = (
            val_predictions * std.reshape(1, 1, -1)
            + mean.reshape(1, 1, -1)
        )
        val_targets = (
            val_targets * std.reshape(1, 1, -1)
            + mean.reshape(1, 1, -1)
        )
        val_metrics = metric(val_predictions, val_targets)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1,
                    "args": vars(args),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                model_path,
            )
            early_stopping_counter = 0
            logging.info("Saved new best model: %s", model_path)
        else:
            early_stopping_counter += 1

        scheduler.step()
        logging.info(
            "Epoch %d/%d | elapsed %.2fs | train_loss %.6f | val_loss %.6f | "
            "val_RSE %.4f | val_RAE %.4f | val_CORR %.4f",
            epoch + 1,
            args.epochs,
            time.time() - epoch_start,
            avg_train_loss,
            avg_val_loss,
            val_metrics["RSE"],
            val_metrics["RAE"],
            val_metrics["CORR"],
        )

        if early_stopping_counter >= args.early_stopping_patience:
            logging.info(
                "Early stopping at epoch %d after %d non-improving epochs",
                epoch + 1,
                args.early_stopping_patience,
            )
            break

    logging.info(
        "Training finished: best_epoch=%d, best_val_loss=%.6f, elapsed=%.2fs",
        best_epoch,
        best_val_loss,
        time.time() - run_start,
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        load_model_state(unwrap_model(model), checkpoint["model_state_dict"])
        logging.info("Loaded best model for testing: %s", model_path)

    logging.info("Testing started")
    test_start = time.time()
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=pin_memory)
            prediction = model(batch_x)
            test_predictions.append(prediction.cpu().numpy())
            test_targets.append(batch_y.permute(0, 2, 1).numpy())

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    test_predictions = (
        test_predictions * std.reshape(1, 1, -1)
        + mean.reshape(1, 1, -1)
    )
    test_targets = (
        test_targets * std.reshape(1, 1, -1)
        + mean.reshape(1, 1, -1)
    )
    test_metrics = metric(test_predictions, test_targets)
    test_time = time.time() - test_start

    logging.info(
        "Testing finished | elapsed %.2fs | RSE %.4f | RAE %.4f | CORR %.4f",
        test_time,
        test_metrics["RSE"],
        test_metrics["RAE"],
        test_metrics["CORR"],
    )

    results = {
        "dataset": args.data,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "num_scales": args.num_scales,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "seed": args.seed,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "norm_type": args.norm_type,
        "dataset_sizes": {
            "train": train_size,
            "validation": val_size,
            "test": test_size,
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "training_time": time.time() - run_start,
        "test_time": test_time,
        "model_path": model_path,
    }
    result_path = save_results(
        results_dir,
        results,
        args.data,
        args.pred_len,
        timestamp,
    )
    logging.info("Results saved to %s", result_path)
    return results


def build_parser():
    """Create the command-line parser for model runs."""
    parser = argparse.ArgumentParser(description="Train and test ASDGC")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset name, such as exchange_rate, illness, PEMS04, PEMS07, solar-energy, ETTm1, ETTm2, or weather.",
    )
    parser.add_argument("--seq_len", type=int, default=32, help="Input sequence length")
    parser.add_argument("--pred_len", type=int, default=24, help="Prediction length")
    parser.add_argument("--num_scales", type=int, default=5, help="Number of temporal scales")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=25,
        help="Early-stopping patience",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight-decay coefficient",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="global",
        choices=["global", "revin", "global_revin"],
        help="Normalization mode selected for the dataset.",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[],
        help="GPU indices, for example: --gpus 0 1",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to a pretrained .pth checkpoint",
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Restore optimizer, scheduler, and epoch state from the checkpoint",
    )
    return parser


def configure_logging(args, timestamp):
    """Configure English file and console logs."""
    log_filename = (
        f"{args.data}_ASDGC_{args.norm_type}_seqlen_{args.seq_len}_"
        f"horizon_{args.pred_len}_{timestamp}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(console)


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configure_logging(arguments, current_timestamp)
    train_model(arguments, current_timestamp)
