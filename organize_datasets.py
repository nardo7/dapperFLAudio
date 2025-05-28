import os
import shutil
from pathlib import Path
import random


def setup_directories(base_path, dataset_name):
    """Create train and test directories for a dataset"""
    train_dir = os.path.join(base_path, dataset_name, "train")
    test_dir = os.path.join(base_path, dataset_name, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return train_dir, test_dir


def organize_crema_d(base_path):
    """Organize CREMA-D dataset by participant ID"""
    print("Organizing CREMA-D dataset...")

    # Setup directories
    train_dir, test_dir = setup_directories(base_path, "CREMA-D")

    # Get all wav files
    source_dir = os.path.join(base_path, "CREMA-D", "AudioWAV")
    wav_files = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

    # Group files by participant ID
    participant_files = {}
    for file in wav_files:
        participant_id = file.split("_")[0]
        if participant_id not in participant_files:
            participant_files[participant_id] = []
        participant_files[participant_id].append(file)

    # Split participants into train and test
    participant_ids = list(participant_files.keys())
    random.shuffle(participant_ids)
    split_idx = int(len(participant_ids) * 0.8)
    train_participants = participant_ids[:split_idx]
    test_participants = participant_ids[split_idx:]

    # Move files to appropriate directories
    for participant_id in train_participants:
        for file in participant_files[participant_id]:
            src = os.path.join(source_dir, file)
            dst = os.path.join(train_dir, file)
            shutil.copy2(src, dst)

    for participant_id in test_participants:
        for file in participant_files[participant_id]:
            src = os.path.join(source_dir, file)
            dst = os.path.join(test_dir, file)
            shutil.copy2(src, dst)

    print(
        f"CREMA-D: {len(train_participants)} participants in train, {len(test_participants)} in test"
    )


def organize_emodb(base_path):
    """Organize EMO-DB dataset by participant ID"""
    print("Organizing EMO-DB dataset...")

    # Setup directories
    train_dir, test_dir = setup_directories(base_path, "EMO-DB")

    # Get all wav files
    source_dir = os.path.join(base_path, "EMO-DB", "wav")
    wav_files = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

    # Group files by participant ID
    participant_files = {}
    for file in wav_files:
        participant_id = file[:2]  # Take first two characters as participant ID
        if participant_id not in participant_files:
            participant_files[participant_id] = []
        participant_files[participant_id].append(file)

    # Split participants into train and test
    participant_ids = list(participant_files.keys())
    random.shuffle(participant_ids)
    split_idx = int(len(participant_ids) * 0.8)
    train_participants = participant_ids[:split_idx]
    test_participants = participant_ids[split_idx:]

    # Move files to appropriate directories
    for participant_id in train_participants:
        for file in participant_files[participant_id]:
            src = os.path.join(source_dir, file)
            dst = os.path.join(train_dir, file)
            shutil.copy2(src, dst)

    for participant_id in test_participants:
        for file in participant_files[participant_id]:
            src = os.path.join(source_dir, file)
            dst = os.path.join(test_dir, file)
            shutil.copy2(src, dst)

    print(
        f"EMO-DB: {len(train_participants)} participants in train, {len(test_participants)} in test"
    )


def organize_ravdess(base_path):
    """Organize RAVDESS dataset by actor ID"""
    print("Organizing RAVDESS dataset...")

    # Setup directories
    train_dir, test_dir = setup_directories(base_path, "RAVDESS")

    # Get all actor directories
    source_dir = os.path.join(base_path, "RAVDESS")
    actor_dirs = [d for d in os.listdir(source_dir) if d.startswith("Actor_")]

    # Split actors into train and test
    random.shuffle(actor_dirs)
    split_idx = int(len(actor_dirs) * 0.8)
    train_actors = actor_dirs[:split_idx]
    test_actors = actor_dirs[split_idx:]

    # Move files to appropriate directories
    for actor_dir in train_actors:
        actor_path = os.path.join(source_dir, actor_dir)
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                src = os.path.join(actor_path, file)
                dst = os.path.join(train_dir, file)
                shutil.copy2(src, dst)

    for actor_dir in test_actors:
        actor_path = os.path.join(source_dir, actor_dir)
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                src = os.path.join(actor_path, file)
                dst = os.path.join(test_dir, file)
                shutil.copy2(src, dst)

    print(f"RAVDESS: {len(train_actors)} actors in train, {len(test_actors)} in test")


def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Base path to the data directory
    base_path = "data"

    # Organize both datasets
    organize_crema_d(base_path)
    # organize_ravdess(base_path)
    # organize_emodb(base_path)

    print("Dataset organization complete!")


if __name__ == "__main__":
    main()
