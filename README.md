# MNIST Digit Classification with PyTorch

A comparison of five neural network architectures for handwritten digit classification. The experiments start with a fully connected baseline and progressively introduce convolution, ReLU, another hidden layer, and dropout.

The best observed test accuracy was **99.49%**, achieved by the convolutional model with two 1,000-neuron hidden layers and dropout.

## Architectures

Each mode selects an architecture and is trained independently from scratch. FC sizes below describe hidden layers; every model ends with a separate 10-output classification layer.

| Mode | Architecture | Hidden activation | Learning rate | Epochs |
| --- | --- | --- | --- | --- |
| 1 | Flatten image → FC(100) → output | Sigmoid | 0.1 | 60 |
| 2 | Two convolution/pooling stages → FC(100) → output | Sigmoid | 0.1 | 60 |
| 3 | Two convolution/pooling stages → FC(100) → output | ReLU | 0.03 | 60 |
| 4 | Two convolution/pooling stages → FC(100) → FC(100) → output | ReLU | 0.03 | 60 |
| 5 | Two convolution/pooling stages → FC(1000) → FC(1000) → output; dropout after each hidden FC activation | ReLU | 0.03 | 40 |

Each convolution uses 40 filters, a 5×5 kernel, stride 1, and no padding. Activation follows each convolution, then 2×2 max pooling with stride 2. The spatial dimensions are:

```text
28×28 → convolution → 24×24 → pooling → 12×12 → convolution →  8×8 → pooling →  4×4
```

The final 40 feature maps are flattened into 640 features before the fully connected layers. Mode 1 instead flattens the original image into 784 pixels. Mode 5 uses dropout with probability 0.5 during training; dropout is disabled during testing.

## Data and training

- **Dataset:** MNIST, with 60,000 training and 10,000 test images of size 28×28, labeled 0–9.
- **Preprocessing:** convert to tensors and normalize with mean 0.1307 and standard deviation 0.3081.
- **Optimizer:** SGD with batch size 10, no momentum, and no weight decay.
- **Loss:** cross-entropy applied to raw output scores (logits).
- **Evaluation:** test loss and accuracy after every epoch, with gradient computation disabled.
- **Device selection:** CUDA first, then Apple MPS if available, then CPU.

## Setup

From the repository root, create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib
```

The dependencies above cover the imports used by the project. `requirements.txt` contains the original pinned environment snapshot.

MNIST is downloaded automatically into `data/` on the first run. Run commands from the repository root so data, logs, and plots are written there.

## Usage

It is suggested to start with a one-epoch check:

```bash
python -u code/train_evaluate_CNN.py --mode 1 --learning_rate 0.1 --num_epochs 1 --batch_size 10
```

For the full comparison, run the following in Bash. The group sends all five runs to the terminal and a single `output.txt`:

```bash
set -o pipefail
{
    python -u code/train_evaluate_CNN.py --mode 1 --learning_rate 0.1 --num_epochs 60 --batch_size 10 &&
    python -u code/train_evaluate_CNN.py --mode 2 --learning_rate 0.1 --num_epochs 60 --batch_size 10 &&
    python -u code/train_evaluate_CNN.py --mode 3 --learning_rate 0.03 --num_epochs 60 --batch_size 10 &&
    python -u code/train_evaluate_CNN.py --mode 4 --learning_rate 0.03 --num_epochs 60 --batch_size 10 &&
    python -u code/train_evaluate_CNN.py --mode 5 --learning_rate 0.03 --num_epochs 40 --batch_size 10
} 2>&1 | tee output.txt
```

Configure the settings explicitly as shown above.

## Outputs

The script prints the run settings, training/test loss and accuracy for each epoch, and the best test accuracy with its epoch. It saves two plots per mode:

```text
plots/
    mode_1_loss.png
    mode_1_accuracy.png
    ...
    mode_5_loss.png
    mode_5_accuracy.png
```

Each plot includes both training and test curves. Rerunning a mode overwrites its plots. Model weights and metric histories are not saved as checkpoints and an interrupted run cannot resume training.

## Results

These results come from one run per architecture on UCF's Newton cluster using an NVIDIA Tesla V100 32 GB GPU and PyTorch 2.14.1 with CUDA 12.6.

| Mode | Final test accuracy | Best test accuracy | Best epoch |
| --- | --- | --- | --- |
| 1 | 97.83% | 97.90% | 34 |
| 2 | 99.06% | 99.16% | 24 |
| 3 | 99.35% | 99.37% | 11 |
| 4 | 99.35% | 99.40% | 6 |
| 5 | **99.46%** | **99.49%** | 27 |

- **Convolution provided the largest gain.** Mode 2 improved best accuracy by 1.26 percentage points over the fully connected baseline, reducing test errors from 210 to 84.
- **The ReLU configuration improved accuracy further.** Mode 3 reached 99.37%, although its learning rate also changed, so the experiment does not isolate activation choice.
- **An additional small FC layer offered little benefit.** Modes 3 and 4 finished with identical test accuracy; their best results differed by only three test images.
- **The wider model with dropout performed best.** Mode 5 misclassified 51 of 10,000 test images at its best epoch. Width, dropout, and training duration changed together, so this result cannot be attributed to dropout alone.
- **Longer training did not always improve generalization.** Training loss approached zero while test loss eventually flattened or increased. Cross-entropy can rise even with stable accuracy when incorrect predictions become more confident.

Best accuracy is the maximum observed across repeated evaluations on the same test set. Each architecture was trained only once, so small differences should be interpreted cautiously and reruns may differ.

## Repository structure

```text
code/
    ConvNet.py             # Layer definitions and five forward passes
    train_evaluate_CNN.py  # MNIST loading, training, evaluation, and plotting
requirements.txt           # Original pinned dependency snapshot
README.md
```
