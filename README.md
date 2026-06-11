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
    --path_img_input examples/input.png \
    --inversion_update_steps 600 \ 
    --path_latents_update "save/weight" \
    --path_tokens_update "save/weight" 

```

### Step 2: Image Editing

Apply text-guided edits using the inverted latent.

```bash
python edit.py \
    --path_latents_update "save/weight" \
    --path_tokens_update "save/weight" \
    --caption 'A meerkat wrapped in a blue towel A*' \
    --edit_caption 'A lion wrapped in a blue towel A*'
    --path_imgs_p2p "save/edit"
```


---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{wu2026tfinv,
  title   = {Training-Free Image Inversion for One-Step Diffusion Models},
  author  = {Wu, Tao and Li, Senmao and Wang, Yaxing and Yang, Shiqi and Wang, Kai and van de Weijer, Joost},
  journal = {Pattern Recognition},
  volume  = {180},
  pages   = {114063},
  year    = {2026},
  doi     = {10.1016/j.patcog.2026.114063}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
