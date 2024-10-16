# Semi-supervised-learning-VAE

Introduction

This project explores the use of a VAE for feature extraction and an SVM for classification, particularly in semi-supervised settings. The VAE learns a latent representation of the Fashion MNIST data, and this latent space is used to train an SVM classifier. The implementation evaluates the performance of this VAE-SVM approach with varying amounts of labeled data.

Data Preparation

The Fashion MNIST dataset is loaded and split into labeled and unlabeled subsets. The labeled subsets are used to train the SVM, while the entire dataset is used for training the VAE. We ensure that each labeled subset contains an equal number of examples from each class. The dataset is normalized and transformed into tensors using the torchvision library.

Model Architecture

The implemented VAE consists of:

	•	Encoder: Two fully connected layers with Softplus activation functions, which map the input images to a latent space. The encoder outputs the mean and log variance for the latent space, which follows a Gaussian distribution.
	•	Reparameterization: A step that samples from the latent space using the mean and log variance to allow backpropagation through the stochastic layer.
	•	Decoder: Two fully connected layers with Softplus activations. The final layer uses a Sigmoid activation to output a probability distribution over pixel values, reconstructing the input from the latent space.

Loss Function

The VAE is trained using a combination of binary cross-entropy (BCE) and Kullback-Leibler divergence (KLD) as the loss function.

Experiment Setup

The experiments involve training the VAE-SVM model with varying amounts of labeled data: 100, 600, 1000, and 3000 labels. A fixed random seed (np.random.seed(42)) ensures reproducibility.

The encoder’s latent space outputs are used to train the SVM classifier. We experimented with different latent space dimensions and SVM kernels, and we found that a 10-dimensional latent space and an RBF (Radial Basis Function) kernel yielded the best results.

Training

To train the VAE and SVM:

	1.	VAE Training:
	•	The VAE is trained on both labeled and unlabeled data. The number of labels can be specified in the run configuration.
	•	During training, the VAE learns to extract latent representations from the images.
	2.	SVM Training:
	•	After training the VAE, the encoder’s output for the labeled data is extracted and used to train the SVM classifier.
	•	The SVM is trained using different kernel functions, with the RBF kernel providing the best classification results.

Training Parameters

	•	Latent Space Dimension: 10
	•	SVM Kernel: RBF (Radial Basis Function)
	•	Labels: 100, 600, 1000, 3000

Results

The table below summarizes the test accuracy of the VAE-SVM model for different numbers of labeled examples:

Number of Labels	Latent Dimension	SVM Kernel	Test Accuracy (%)
100	10	RBF	64.82
600	10	RBF	76.32
1000	10	RBF	76.47
3000	10	RBF	79.90

From the results, it is evident that increasing the number of labeled data samples improves the classification accuracy. Even with a small number of labels, the VAE combined with the RBF kernel SVM can achieve competitive performance.

Training and Testing Instructions

To train and test the VAE-SVM model, follow these steps:

	1.	Training:
	•	Run the training script, specifying the number of labels to use. This will train the VAE, extract latent representations, and train the SVM classifier.
	2.	Testing:
	•	After training, the SVM classifier is evaluated on the test set, and the classification accuracy is displayed for each experiment.
	3.	Saving Weights:
	•	The VAE weights are saved after each experiment in the vae_weights directory for future use.

Conclusion

This project demonstrates the effectiveness of combining a VAE for feature extraction with an SVM classifier for semi-supervised learning on the Fashion MNIST dataset. The results show that the VAE-SVM model performs well, even with limited labeled data.

References

	•	Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014). Semi-supervised Learning with Deep Generative Models. Advances in Neural Information Processing Systems.
