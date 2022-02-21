# Appendix
## 4.1.1. Creating the the Digit ‘Չ’ using MoVAE
### Dataset
The dataset is downloaded from https://keras.io/api/datasets/mnist/. Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset. We use 60000 training images and 10000 testing images as split exactly by the dataset. 

We use a batch size of 32, so the entire dataset is batched to train the network in the following training configuration. 

### Base VAE	

![Base VAE Network](./image_assets/4.1.1_Base_VAE_Network.png?raw=true)

Layer structure of base VAE network. (32, (5,5), (2,2), s) indicates that this two dimensional convolutional layer has 32 filters, 5x5 kernel size, 2x2 strides and padding parameter ‘same’. The base VAE training uses Adam optimizer with a learning rate of 1e-6. It is trained for 150 epochs. We follow the exact definition of the reparameterization trick and loss function as the original VAE paper suggested (Kingma and Welling 2014). 
	
### Auxiliary Discriminator 

![Auxiliary Discriminator Network](./image_assets/4.1.1_Auxiliary_Discriminator_Network.png?raw=true)

Layer structure of Auxiliary discriminator. (64, (5,5), (2,2), s) indicates that this two dimensional convolutional layer has 64 filters, 5x5 kernel size, 2x2 strides and padding parameter ‘same’. We train the auxiliary discriminator with 20 epochs using Adam optimizer and a learning rate of 1e-4. The loss is the categorical cross entropy defined as in the Tensorflow keras library. 
	
### MoVAE
The loss MoVAE is defined as 

<img src="https://i.upmath.me/svg/L_%7BMoVAE%7D%20%3D%20L_%7BVAE%7D%20%2B%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%5Calpha_%7Bk%7D%20L_%7BD_%7Bk%7D%7D" alt="L_{MoVAE} = L_{VAE} + \sum_{k = 1}^{K} \alpha_{k} L_{D_{k}}" /> 

where <img src="https://i.upmath.me/svg/L_%7BVAE%7D" alt="L_{VAE}" /> = base VAE loss, <img src="https://i.upmath.me/svg/K" alt="K" />= number of auxiliary discriminators, <img src="https://i.upmath.me/svg/%5Calpha" alt="\alpha" />= weight coefficient, <img src="https://i.upmath.me/svg/L_%7BD_%7Bk%7D%7D" alt="L_{D_{k}}" />= auxiliary discriminator loss. We always use Adam optimizer with a learning rate of 1e-6. And the value is 10000 for each objective. The MoVAE network is trained for 20 epochs. 

## 4.1.2. Creating the Digit ‘Չ’ using MoGAN 
### Dataset
The dataset is downloaded from https://keras.io/api/datasets/mnist/. Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset. We use 60000 training images and 10000 testing images as split exactly by the dataset. 

We use a batch size of 32, so the whole dataset is batched to train the network in the following training configuration. 

### Base GAN

![Base GAN Network](./image_assets/4.1.2_Base_GAN_Network.png?raw=true)

Layer structure of base GAN network. (64, (5,5), (2,2), s) indicates that this two dimensional convolutional layer has 32 filters, 5x5 kernel size, 2x2 strides and padding parameter ‘same’. The base GAN training uses Adam optimizer with a learning rate of 1e-6. It is trained for 150 epochs. We follow the exact definition of the loss function as the original GAN paper suggests (I. J. Goodfellow et al. 2014). 
	
### Auxiliary Discriminator 

![Auxiliary Discriminator Network](./image_assets/4.1.2_Auxiliary_Discriminator_Network.jpg?raw=true)

Layer structure of Auxiliary discriminator. (64, (5,5), (2,2), s) indicates that this two dimensional convolutional layer has 64 filters, 5x5 kernel size, 2x2 strides and padding parameter ‘same’. We train the auxiliary discriminator with 20 epochs using Adam optimizer and a learning rate of 1e-4. The loss is the categorical cross entropy defined as in the Tensorflow keras library. 
	
### MoGAN

The loss MoGAN is defined as 

<img src="https://i.upmath.me/svg/L_%7BMoGAN%7D%20%3D%20L_%7BD_%7BR%7D%7D%20%2B%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%5Calpha_%7Bk%7D%20L_%7BD_%7Bk%7D%7D" alt="L_{MoGAN} = L_{D_{R}} + \sum_{k = 1}^{K} \alpha_{k} L_{D_{k}}" />

where <img src="https://i.upmath.me/svg/L_%7BD_%7BR%7D%7D" alt="L_{D_{R}}" /> = realistic discriminator loss. <img src="https://i.upmath.me/svg/K" alt="K" />= number of auxiliary discriminators, <img src="https://i.upmath.me/svg/%5Calpha" alt="\alpha" />= weight coefficient, <img src="https://i.upmath.me/svg/L_%7BD_%7Bk%7D%7D" alt="L_{D_{k}}" />= auxiliary discriminator loss. To train MoGAN, we use Adam optimizer with a learning rate of 1e-4 for realistic discriminator and same parameter for the generator. The value is 10 for each objective. The MoGAN network is trained for 20 epochs. 

## 4.2. Chair Design Using MoVAE
### Dataset
We begin by obtaining all available chair geometries from the ShapeNet Database https://shapenet.org/ (Wu et al. 2016). Specifically we use the ShapeNetCore which contains 6778 geometries under the name ‘chair’. However, the dataset also includes sofas, and a relatively small number of toilet seats and benches. We write and execute a custom grasshopper script that converts geometris to solid voxels with dimensions of 32x32x64. 

Due to the lack of needed data labels, ratings of each criterion are done by one researcher in our team. Stability factor is a continuous variable and is calculated as  the ratio of the moment of overturning generated by a horizontal force and the self weight of the chair. A sample of ratings are shown in Figure Appendix 4.2.1, flimsy chairs have ratings near zero on one end of the spectrum, and more boxy, sofa-like chairs on the other. The function factor is a categorical variable rated by one of the researchers. In everyday life, a complex relationship of factors influences a human observer’s assessment of a chair being a leisure or a work chair. In our ratings, factors such as the angle of the backrest, presence of wheels or cushions, perceived softness, and thickness of the seat were taken into account. The researcher is asked to give a rating from 0 to 10, where 0 means the chair is used most certainly for leisure and recreation and 10 means the chair is used mostly for work, requiring long periods of sedentary activity. A sample of ratings are shown in Figure Appendix 4.2.2. The aesthetic preference factor is a categorical variable rated by one of the researchers. This researcher is asked to rate the chairs on a scale from 0 to 10 according to personal aesthetic preference. Certainly a person’s aesthetic preference is based on complex relationships of various aspects and is influenced by personal experiences. However, for the purpose of this project, we will only take into consideration this researcher’s rating. The researcher reflected a personal tendency to rate more organically formed “curvy” chairs with integrated seats and backrest with a high number, while chunky and “boxy” chairs were rated with a low number. A sample of ratings is illustrated in Figure Appendix 4.2.3. 

We use a batch size of 32, so the whole dataset is batched to train the network in the following training configuration. 

![A sample of ratings for the stability factor](./image_assets/4.2_Sample_ratings_stability_factor.png?raw=true)

Figure Appendix 4.2.1. A sample of ratings for the stability factor. 

![A sample of ratings for the leisure / work factor](./image_assets/4.2_Sample_ratings_leisurework_factor.png?raw=true)

Figure Appendix 4.2.2. A sample of ratings for the leisure / work factor. 

![A sample of ratings for the aesthetic preference factor](./image_assets/4.2_Sample_ratings_aesthetic_factor.png?raw=true)

Figure Appendix 4.2.3. A sample of ratings for the aesthetic preference factor. 

### Base VAE

![Base VAE Network](./image_assets/4.2_Base_VAE_Network.png?raw=true)

Layer structure of base VAE network. (32, (5,5,5), (1,1,1), s) indicates that this three dimensional convolutional layer has 32 filters, 5x5x5 kernel size, 1x1x1 strides and padding parameter ‘same’. The base VAE training uses Adam optimizer with a learning rate of 1e-4. It is trained for 100 epochs. And we follow the exact definition of the reparameterization trick and loss function as the original VAE paper suggested (Kingma and Welling 2014). 
	
### Auxiliary Discriminator 

![Auxiliary Discriminator Network](./image_assets/4.2_Stability&Aesthetic_Auxiliary_Discriminator_Network.png?raw=true)

Layer structure of stability and aesthetic preference auxiliary discriminators. (16, (4,4,4), (2,2,3), s) indicates that this three dimensional convolutional layer has 16 filters, 5x5x5 kernel size, 2x2x2 strides and padding parameter ‘same’. 
For the stability discriminator, we train the network with 50 epochs using Adam optimizer and a learning rate of 1e-2. The loss is the sum of mean squared errors defined as in the Tensorflow keras library. 
For the aesthetic preference discriminator, we train the network with 100 epochs using Adam optimizer and a learning rate of 1e-2. The loss is the sum of mean squared errors  defined as in the Tensorflow keras library. 

![Auxiliary Discriminator Network](./image_assets/4.2_LeisureWork_Auxiliary_Discriminator_Network.png?raw=true)

Layer structure of leisure / work auxiliary discriminators. For this discriminator, we train the network with 100 epochs using Adam optimizer and a learning rate of 1e-2. The loss is the sum of mean squared errors  defined as in the Tensorflow keras library. 
	
### MoGAN
The loss MoGAN is defined as 

<img src="https://i.upmath.me/svg/L_%7BMoGAN%7D%20%3D%20L_%7BD_%7BR%7D%7D%20%2B%20%5Csum_%7Bk%20%3D%201%7D%5E%7BK%7D%20%5Calpha_%7Bk%7D%20L_%7BD_%7Bk%7D%7D" alt="L_{MoGAN} = L_{D_{R}} + \sum_{k = 1}^{K} \alpha_{k} L_{D_{k}}" />

where <img src="https://i.upmath.me/svg/L_%7BD_%7BR%7D%7D" alt="L_{D_{R}}" /> = realistic discriminator loss. <img src="https://i.upmath.me/svg/K" alt="K" />= number of auxiliary discriminators, <img src="https://i.upmath.me/svg/%5Calpha" alt="\alpha" />= weight coefficient, <img src="https://i.upmath.me/svg/L_%7BD_%7Bk%7D%7D" alt="L_{D_{k}}" />= auxiliary discriminator loss. To train MoGAN, we use Adam optimizer with a learning rate of 1e-6. And the values are 50000, 60000, and 60000 for stability objective, leisure / work objective and aesthetic appearance objective respectively. The MoVAE network is trained for 20 epochs. For the results obtained in this paper, we set low stability target to 0.05, high stability target to 0.55; low leisure / work target to 0.05, high leisure / work target to 0.85; low aesthetic preference target to 0.05, high aesthetic preference target to 0.75. 
