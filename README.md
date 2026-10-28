---
title: CrackVision
emoji: 🧱
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Concrete crack detection with MobileNetV2 + Grad-CAM
---

# CrackVision: Concrete Crack Detection

Upload or take a photo of a concrete surface. The app splits it into 227×227 patches,
classifies each patch as **Crack / No Crack**, and highlights cracked areas with Grad-CAM.

- Model: MobileNetV2 (transfer learning, fine-tuned)
- Data: Concrete Crack Images for Classification (Özgenel, 2018), 40,000 images
- Team: Natthiwut Phimtakhlong, Chakrit Rodmanee, Thongtong Phomniyomkit
