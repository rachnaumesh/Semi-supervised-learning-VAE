# Semi-supervised-learning-VAE

## Introduction

This project implements a VAE as a feature extractor and uses an SVM for classification on the Fashion MNIST dataset. The implementation includes:

- Training the VAE with varying amounts of labeled data
- Using the VAE as a feature extractor for the SVM classifier
- Evaluating the model's performance with different numbers of labels

## Data Preparation

The Fashion MNIST dataset is used for this project. The data is split into labeled and unlabeled subsets, with the labeled subset containing equal numbers of examples from each class.

## Model Architecture

The VAE architecture consists of:

1. Encoder:
   - Two fully connected hidden layers with Softplus activations
   - Outputs mean and log variance for the latent space

2. Reparameterization step:
   - Allows backpropagation through the stochastic layer
   - Samples from the latent space using mean and log variance

3. Decoder:
   - Two hidden layers with Softplus activations
   - Final layer uses sigmoid activation

## Training

Set random seed for reproducibility:

np.random.seed(42)

## Loss Function

The VAE is trained using a combination of **Binary Cross-Entropy (BCE)** and **Kullback-Leibler Divergence (KLD)** as the loss function.

## Training Steps

1. Run the `run_experiment` function with the desired number of labels.
   - This function trains the VAE, extracts latent representations, and fits the SVM classifier.

## Training Parameters

- **Number of labeled examples**: 100, 600, 1000, and 3000
- **Latent space dimension**: 10
- **SVM kernel**: RBF (Radial Basis Function)

## Results

The model's performance is evaluated using test accuracy for different numbers of labeled examples:

| Number of Labels | Latent Dimension | SVM Kernel | Test Accuracy (%) |
|------------------|------------------|------------|-------------------|
| 100              | 10               | RBF        | 64.82             |
| 600              | 10               | RBF        | 76.32             |
| 1000             | 10               | RBF        | 76.47             |
| 3000             | 10               | RBF        | 79.90             |

## Testing

To test the model:

- The SVM is evaluated on the test set after training.
- Classification accuracy is displayed for each experiment.

## Saving Weights

- The VAE weights are saved after each experiment in the `vae_weights` directory for future reference.

## Conclusion

This project demonstrates the implementation of a semi-supervised learning approach using a VAE and SVM on the Fashion MNIST dataset. The results show the effectiveness of this method in achieving competitive performance with limited labeled data.

## References

- Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014). *Semi-supervised Learning with Deep Generative Models*. Advances in Neural Information Processing Systems, 27.
