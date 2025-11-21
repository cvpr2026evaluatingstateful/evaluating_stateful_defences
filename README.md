# ⭐ DazzlePatch Attack on AdvQDet

This repository provides scripts to run our novel **DazzlePatch Attack**, as well as the **Square Attack** proposed by *Andriushchenko et al.* (2019), against **AdvQDet**, the adversarial detector introduced by *Wang et al.* (2024).

Our implementation is built on top of the official AdvQDet codebase.

👉 **Original AdvQDet repository:** https://github.com/xinwong/AdvQDet

---

## 📦 Installation

Install all required packages with:

    pip install -r requirements.txt

You may also need to install additional dependencies listed in the **official AdvQDet repository**.

---

## 🚀 Running the Attacks

### 🔥 DazzlePatch Attack

To run the **DazzlePatch Attack** on AdvQDet, execute:

    python hightexture_snowbank_patchattack_imagenet.py

This script applies our DazzlePatch perturbation on selected ImageNet samples and evaluates the robustness of AdvQDet.

---

### 🟥 Square Attack

To run the **Square Attack** (*Andriushchenko et al.*, 2019) on AdvQDet, execute:

    python hightexture_snowbank_patchattack_imagenet(squareattackonly).py

This script evaluates AdvQDet under the Square Attack in our experimental setting.

---
