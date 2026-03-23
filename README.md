# 🚀 Training-Free Image Inversion for One-Step Diffusion Models

> Official implementation of the paper  
> **Training-Free Image Inversion for One-Step Diffusion Models**

---

## 📦 Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/tttao-uwu/TFinv.git
cd TFinv

# Create conda environment
conda create -n tf_inversion python=3.10
conda activate tf_inversion

# Install dependencies
pip install -r requirements.txt

---

## 🧪 Example Workflow

# Step 1: Image Inversion
python inversion.py \
    --image_path examples/input.png \
    --output_dir results/inversion

# Step 2: Image Editing
python edit.py \
    --latent_path results/inversion/latent.pt \
    --prompt "a cyberpunk style portrait" \
    --output_dir results/edit



