# fashion-mnist-autoencoder
Convolutional Autoencoder built in TensorFlow/Keras for 2D bottleneck compression and image reconstruction on Fashion-MNIST

# 👖 Fashion-MNIST Convolutional Autoencoder

An end-to-end TensorFlow/Keras implementation of a Convolutional Autoencoder that compresses $32 \times 32$ grayscale clothing images down to a **2-dimensional continuous latent space** (a $99.8\%$ compression bottleneck) and reconstructs them back.

---

## 🎯 Reconstructed Image Results

Original vs. Reconstructed images across 5 representative clothing categories (Ankle Boot, Trouser, Bag, Dress, Pullover):

![Reconstruction Results](assets/reconstruction.png)

---

## 🏗️ Model Architecture

The model consists of a symmetric Convolutional Encoder-Decoder pair:

* **Encoder:**
  * `Conv2D` (32 filters, $3 \times 3$, stride 2, ReLU)
  * `Conv2D` (64 filters, $3 \times 3$, stride 2, ReLU)
  * `Conv2D` (128 filters, $3 \times 3$, stride 2, ReLU)
  * `Flatten` $\rightarrow$ `Dense` (2 units, continuous bottleneck)
* **Decoder:**
  * `Dense` $\rightarrow$ `Reshape`
  * `Conv2DTranspose` (128 filters, $3 \times 3$, stride 2, ReLU)
  * `Conv2DTranspose` (64 filters, $3 \times 3$, stride 2, ReLU)
  * `Conv2DTranspose` (32 filters, $3 \times 3$, stride 2, ReLU)
  * `Conv2D` (1 filter, $3 \times 3$, Sigmoid output)

* **Loss Function:** Binary Cross-Entropy
* **Optimizer:** Adam

---

## 🔬 Key Insights & Reconstruction Discrepancies

1. **Loss of High-Frequency Details:** Fine structures (e.g., shoe heels, handbag straps, text logos like "Lee") disappear during reconstruction because a 2D bottleneck cannot preserve high-frequency spatial variation.
2. **Boundary Blurring:** Sharp edges (e.g., trouser leg separations) become continuous gradient blurs due to pixel-wise Binary Cross-Entropy penalization averaging out spatial uncertainty.
3. **Global Geometry Retention:** Global shapes, such as trouser leg outlines or sweater torso shapes, survive compression well because low-dimensional coordinates capture basic height and width features.

---

## 🚀 Future Enhancements

* Implement a **Variational Autoencoder (VAE)** with KL-divergence loss to enforce a continuous normal distribution in the latent space.
* Increase latent bottleneck dimensions (e.g., $d=16$ or $d=32$) to evaluate detail retention trade-offs.
