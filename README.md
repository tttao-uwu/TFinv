# 🚀 Training-Free Image Inversion for One-Step Diffusion Models

> Official implementation of the paper
> **Training-Free Image Inversion for One-Step Diffusion Models**

---

## 📦 Installation

Clone the repository and set up the environment:

```bash
# Clone the repo
git clone https://github.com/tttao-uwu/TFinv.git
cd TFinv

# Create conda environment
conda create -n tf_inversion python=3.10
conda activate tf_inversion

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Example Workflow

### Step 1: Image Inversion

Invert an input image into the latent space of the one-step diffusion model.

```bash
python inversion.py \
    --image_path examples/input.png \
    --output_dir results/inversion
```

### Step 2: Image Editing

Apply text-guided edits using the inverted latent.

```bash
python edit.py \
    --latent_path results/inversion/latent.pt \
    --prompt "a cyberpunk style portrait" \
    --output_dir results/edit
```

---

## 📁 Project Structure

```
TFinv/
├── inversion.py        # Image inversion script
├── edit.py             # Image editing script
├── examples/
│   └── input.png       # Example input image
├── results/
│   ├── inversion/      # Inversion outputs
│   └── edit/           # Editing outputs
└── requirements.txt    # Python dependencies
```

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{tfinv2024,
  title     = {Training-Free Image Inversion for One-Step Diffusion Models},
  author    = {Your Name},
  year      = {2024},
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
