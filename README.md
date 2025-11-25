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

## ❄️📥 Downloading Open Images (Snow Category)

This project uses image references from the Open Images Dataset v7.
The required snow-related images can be downloaded using the script provided.

### ⬇️ Download Snow Images

Run the following command:

    python openimagev7_imagesdownloaded_fromimageIDs.py snow_image_ids.txt --download_folder=openimages_snow

### 📂 What This Command Does

Reads image IDs from snow_image_ids.txt

Fetches the corresponding image URLs from the Open Images metadata

Downloads all images into:

    openimages_snow/
## 📥 Downloading ImageNet Validation Images

The link to the images is originally provided at https://github.com/TransEmbedBA/TREMBA

Please download the test images from https://drive.google.com/file/d/1Gs_Rw-BDwuEn5FcWigYP5ZM9StCufZdP/view?usp=sharing and extract them under data/imagenet/test_images

## 📁 Example ImageNet Test Folder Structure

Below is an example of how your ImageNet test images should be organized after extraction:

    data/
    └── imagenet/
        └── test_images/
            └── Sample_1000/
                └── n04418357/
                    └── ILSVRC2012_val_00021503.JPEG

## 🚀 Running the Attacks

### 🔥 DazzlePatch Attack

To run the **DazzlePatch Attack** on AdvQDet, execute:

    python dazzlepatch_attack_advqdet.py

This script applies our DazzlePatch perturbation on selected ImageNet samples and evaluates the robustness of AdvQDet.

---

### 🟥 Square Attack

To run the **MultiSquare Square Attack** (*Andriushchenko et al.*, 2019) on AdvQDet, execute:

    python multisquareattack_advqdet.py

This script evaluates AdvQDet under the Square Attack in our experimental setting.

---

⚠️ Licensing Notice

Images are not stored in this repository.
They are downloaded from their original sources in compliance with Open Images licensing:

Annotations: CC BY 4.0 (Google LLC)

Images: Listed under CC BY 2.0
