import subprocess, sys

fold = 0
epochs = 8

cmd = [
    sys.executable, "-m", "scripts.train_baseline_densenet",
    "--fold", str(fold),
    "--image_view", "crop",
    "--epochs", str(epochs),
    "--batch_size", "16",
    "--lr", "1e-4",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)