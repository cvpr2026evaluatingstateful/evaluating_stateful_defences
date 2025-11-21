DazzlePatch Attack on AdvQDet

This repository provides scripts to run our novel DazzlePatch Attack, as well as the Square Attack proposed by Andriushchenko et al. (2019), against AdvQDet, the adversarial detector introduced by Wang et al. (2024).

Our implementation is built on top of the official AdvQDet codebase.

Original AdvQDet repository: https://github.com/xinwong/AdvQDet

📦 Installation

Install all required packages using:

pip install -r requirements.txt


You may also need to install additional dependencies from the original AdvQDet repository.

🚀 Running the Attacks
DazzlePatch Attack

To simulate the DazzlePatch attack on AdvQDet, run:

python hightexture_snowbank_patchattack_imagenet.py

To simulate the Square Attack on AdvQDet, run:

python hightexture_snowbank_patchattack_imagenet(squareattackonly).py
