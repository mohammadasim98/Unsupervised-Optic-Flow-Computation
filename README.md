# Multi-Scale based Optic Flow Computation with Deep Energy
### A CNN based computation of Optic Flow by unsupervised means using Multi-Scale Deep Energy Reduction

## Abstract
Advancement in Deep Learning has opened a portal for a variety of applications. Such an application includes image analysis and synthesis that range from image-to-image tasks such as denoising, in-painting, and segmentation to classification with other tasks. In our work, we used the idea of using Deep Energy to train a model for computing optic flow field, given a pair of subsequent frames. This task is relevant in self-driving technology, stereo reconstruction, and image registration. Our work uses variational calculus to design an energy loss function to train a deep learning model without using ground truths. It is similar to the Physics-Informed Neural Networks, where the goal is to use the PDEs to minimize the residual loss and allow the model to train from the given loss by updating its corresponding weights. The dataset consists of synthetic video frames provided by MPI-Sintel generated using Blender software. 

## Introduction
For decades researchers came up with different numerical schemes and algorithms which can compute optic flow given a pair of subsequent frames. These were slower because the algorithm computes the optic flow for each pair of frames in several iterations. They come in the category of local and global techniques with advantages and disadvantages. [7] proposed the first local method where they assumed that the brightness of pixels is equal for both frames given the illumination is constant. With this assumption, we can draw an optic flow constraint that is solved to compute the components of the flow field by using a window. Having a window,  the linear system of equations is overdetermined because we assume the flow fields are equal in the window. Then it is solved via the least square methods.

At the same time, [6] proposed a global technique that uses variational calculus which converts the flow field computation into a minimization task. They derived an energy functional that is minimized to solve for the components of the flow field by using similar assumptions as for the local method with an addition of a smoothness assumption. In such an assumption, the resulting optic flow is smooth and does not have discontinuities. 

These methods use explicit numerical schemes where the algorithm has to be iterated several times for a given pair of frames to compute the respective optic flow. The number of iterations can range from 1000 to 100000 or more. For this reason, implementing it for real-time optic flow computation is cumbersome. So researchers came up with techniques that use deep learning and variational energy to train the model for computing the flow fields. Then the trained model can perform real-time optic flow computation. 

Variational energy as the loss function comes under the domain of unsupervised learning. In the supervised approach, we use the ground truth to model the loss function that performs better than the unsupervised approach in most cases. But there are more benefits to an unsupervised approach. For instance, providing the model with unannotated data is efficient. The model can be updated when deployed in an application. So this allows for continuous learning with newer data and better performance with time.

## Related Works
The method proposed by [7] experienced the aperture problem because of the small window size where flow fields become harder to compute when the linear system of equations is under-determined. To solve this, [6] introduced a global technique using variational energy under the assumption of brightness constancy and a smoother flow field. They used homogeneous diffusion weighted with a parameter alpha to introduce smoothness into the flow field. The smoothness and the data term lead to a diffusion-reaction process. They designed an energy functional that has a data term $E_{data}$ and a smoothness term $E_{smooth}$ as shown in the following.

$$E_{data} = |I(x+u, y+v, z+1) - I(x, y, z)|^2$$

Where $I(x+u, y+v, z+1)$ is the image at time unit $z+1$.

$$E_{smooth} = |\nabla u|^2 + |\nabla v|^2$$

Where $u, v$ are the flow field in the direction of x and y respectively. The final variational energy has the following form.

$$E(u,v) = \int (E_{data} + \alpha E_{smooth}) dxdy$$

Since then, the global variational approach has been the most popular among all other classical methods. There are several modifications to the energy functional where researchers tried different assumptions such as gradient and hessian constancy proposed by [2] and a variety of models for the data and the smoothness terms. It works very well for sequences with smaller displacements but fails for larger displacements. 

For that, [2] added a multi-scale extension to the technique proposed by [6] in which they introduced a coarse-to-fine strategy to compute the optic flow starting from the lowest resolution. The displacements are much smaller at a smaller scale where the variational approach can perform well. After calculating the flow fields at a lower resolution, it is up-sampled by interpolation and then set as initialization for computing optic flow at the finer scale. The steps are repeated till the main scale is achieved.

Since these techniques use explicit numerical schemes, it was still not implemented in neural networks. [4] proposed an encoder-decoder network similar to U-Net termed FlowNet that allows for optic flow computation with higher accuracy. It was one of the first leaps in using a deep neural network model for optic flow computation. They used a supervised approach to train the model but later researchers used the ideas from FlowNet and combined them with a variational approach to train the model. It turned out that with the encoder-decoder style of FlowNet, it is possible to use it for multi-scale optic flow computation. Numerous extensions for FlowNet architecture have since been proposed.

Inspired by these ideas, our work makes the use of variational energy with a non-quadratic penaliser, coarse-to-fine warping, and an advanced smoothness assumption to construct a loss function that may improve the performance of an unsupervised deep learning model. To design the model, we used the ideas from FlowNet architecture and reduced the overall size to make it compatible with the available hardware for training.

## Methodology
In this section, we discuss the implementation of our project. We will first discuss the mathematical modeling for the variational energy followed by the model architecture and approaches we used. Then we will also discuss how the data set was prepared and processed.
### Mathematical Modelling
For modeling the data-term, we used the sum of squared differences (SSD) between the warped and the original frame through the computed optic flow. It is based on the assumption that the brightness of the two subsequent pixels stays the same. We used the albedo version of the MPI-Sintel dataset that satisfies this assumption. The SSD passes through a non-quadratic penaliser proposed by [3] which is as follows.

$$\psi_D (s^2) = \sqrt{\epsilon^2 + s^2}$$

This penaliser is convex and makes the modeling more robust against outliers and noises. Let ${I_0, I_1}$ be the subsequent frames, then our final data term is shown in the following equation.

$$E_{data} = \psi_D(|I_{warped} - I_0|^2)$$

where $I_{warped}$ is the inverse warped version of $I_1$ using the computed optic flow $(u, v)$. We also added a smoothness term similar to the one proposed by [6]. More specifically, we used a linear isotropic image-driven diffusion proposed by [1] which is as follows.

$$E_{smooth} = \psi_S(|\nabla I_0|^2)(|\nabla u|^2 + |\nabla v|^2)$$

where $\psi_s(s^2)$ is a diffusivity function proposed by 
[8] which is as follows.
$$\psi_S(s^2) = \frac{1}{1+ s^2/\lambda^2}$$

This diffusivity is inversely proportional to the gradient magnitude of the frame $I_0$ which means that at larger gradients, the diffusivity is smaller hence edges are preserved. For comparing the effect of using both isotropic and homogeneous modeling for the smoothness term, we set up two experiments on the two smoothness models discussed in section 4.


![alt text](../imgs/flownet.jpg)
Our deep learning model is inspired by FlowNetSimple architecture proposed by \cite{fischer} and the ResNet50 architecture proposed by \cite{he}. Our hybrid model is a shrunk version of the FlowNetSimple and the only extra feature is the residuals added in the encoder part of the network. We reduced the number of convolutional layers and the number of filters for each convolution such that our highest number of channels is 64, whereas, in the FlowNetSimple, the highest number of channels is 1024. We then added multi-scale losses with the help of our down-sampler and the differentiation block. The final loss is computed by summing the multi-scale losses weighted by a monotonically decreasing weighting factor with increasing resolution. This way, we can emphasize more towards the smaller scale which acts as a recipe for computing optic flow at a larger scale in a pyramidal fashion. The complete abstract view of the architecture is shown in figure 1.

For each scale, we compute the energies using the specified data and smoothness terms. The energies are summed for each pixel in an image. The down-sampling block performs average pooling with a kernel size of 2. The input frames and the optic flows at each scale must match their dimensions. So in the reduced FlowNetSimple architecture, we added padding to each convolution/de-convolution layer while performing down-sampling through average pooling to keep the dimension consistent and easier to manage. 

To make the differentiator block efficient, we stacked the \begin{math}(u, v)\end{math} component of the optic flow, and the frames \begin{math}I_0\end{math} and \begin{math}I_1\end{math} depth-wise into tensors for each scale. Then we apply depth-wise convolution with x- and y-directional 3x3 Sobel operators. The output of x- and y-directional Sobel operation is then concatenated depth-wise i.e. let \begin{math}T\end{math} be the input to the differentiator block, then the output will be \begin{math}\{T_x, T_y\}\end{math} for each scale.

To design a relatively small architecture, we started with a bigger version of the modified FlowNetSimple model with more layers and channels. Then by performing several experiments, we shrunk the number of trainable weights from ~7M to ~170K. More specific details regarding the model are described in table 1.
\begin{table*}[htb]
\caption{Reduced FlowNetSimple Configuration}
\label{table:demo}
\begin{tabular}{p{2cm}p{2cm}p{2cm}p{2cm}p{2cm}p{2cm}p{2cm}}
\hline
Activation & Batch Norm & Pooling & Residuals & U-Net-like & Parameters \# & layers (including concat \& diff blocks)\\
\hline
Leaky ReLU & No & Average & Yes & Yes & 171,846 & 92\\
\hline
\end{tabular}
\end{table*}
