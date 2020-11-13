# Deep learning

Slides: [pdf](https://www.tu-chemnitz.de/informatik/KI/edu/deeprl/lectures/pdf/2.7-DeepNetworks.pdf)

## Feedforward neural networks

<div class='embed-container'><iframe src='https://www.youtube.com/embed/pVneu-1EYdI' frameborder='0' allowfullscreen></iframe></div>

An **artificial neural network** (ANN) is a cascade of **fully-connected** (FC) layers of artificial neurons.

```{figure} ../img/shallowdeep.png
---
width: 100%
---
Shallow vs. deep neural networks.
```

Each layer $k$ transforms an input vector $\mathbf{h}_{k-1}$ into an output vector $\mathbf{h}_{k}$ using a weight matrix $W_k$, a bias vector $\mathbf{b}_k$ and an activation function $f()$.

$$\mathbf{h}_{k} = f(W_k \times \mathbf{h}_{k-1} + \mathbf{b}_k)$$

Overall, ANNs are **non-linear parameterized function estimators** from the inputs $\mathbf{x}$ to the outputs $\mathbf{y}$ with parameters $\theta$ (all weight matrices and biases).

$$\mathbf{y} = F_\theta (\mathbf{x})$$

ANNs can be used for both **regression** (continuous outputs) and **classification** (discrete outputs) tasks. In supervised learning, we have a fixed **training set** $\mathcal{D}$ of $N$ samples $(\mathbf{x}_t, \mathbf{t}_i)$, where $t_i$ is the **desired output** or **target**.

* **Regression:**

    * The output layer uses a **linear** activation function: $f(x) = x$

    * The network minimizes the **mean square error** (mse) of the model on the training set:

    $$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{t} \in \mathcal{D}} [ ||\mathbf{t} - \mathbf{y}||^2 ]$$ 


* **Classification:**

    * The output layer uses the **softmax** operator to produce a probabilty distribution: $y_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$

    * The network minimizes the **cross-entropy** or **negative log-likelihood** of the model on the training set:

    $$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{t} \in \mathcal{D}} [ - \mathbf{t} \, \log \mathbf{y} ]$$


The cross-entropy between two probability distributions $X$ and $Y$ measures their similarity:

$$
    H(X, Y) = \mathbb{E}_{x \sim X}[- \log P(Y=x)]
$$

It measures whether samples from $X$ are likely under $Y$? Minimizing the cross-entropy makes the two distributions equal almost anywhere.

```{figure} ../img/crossentropy.svg
---
width: 100%
---
The cross-entropy measures the similarity of the two probability distributions.
```

In supervised learning, the targets $\mathbf{t}$ are fixed **one-hot encoded vectors**.

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{t} \in \mathcal{D}} [ - \sum_j t_j \, \log y_{j} ]$$

But it could be any target distribution.

```{figure} ../img/crossentropy-animation.gif
---
width: 70%
---
Cross-entropy between multinomial distributions.
```

In both cases, we want to minimize the loss variant by applying **Stochastic Gradient Descent** (SGD) or a variant (Adam, RMSprop).

$$
    \Delta \theta = - \eta \, \nabla_\theta \mathcal{L}(\theta)
$$

The question is how to compute the **gradient of the loss function** w.r.t the parameters $\theta$. For both the mse and cross-entropy loss functions, we have:

$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_\mathcal{D} [- (\mathbf{t} - \mathbf{y}) \, \nabla_\theta \, \mathbf{y}]$$

There is an algorithm to compute efficiently the gradient of the output w.r.t the parameters: **backpropagation** (see Neurocomputing). In deep RL, we do not care about backprop: tensorflow or pytorch do it for us. 

```{figure} ../img/deeprl.jpg
---
width: 80%
---
Principle of deep reinforcement learning.
```

There are three aspects to consider when building a neural network:

1. **Architecture:**  how many layers, what type of layers, how many neurons, etc.

    * Task-dependent: each RL task will require a different architecture. Not our focus.

2. **Loss function:** what should the network do?

    * Central to deep RL!

3. **Update rule** how should we update the parameters $\theta$ to minimize the loss function? SGD, backprop.

    * Not really our problem, but see *natural gradients* later.

## Convolutional neural networks

<div class='embed-container'><iframe src='https://www.youtube.com/embed/t244PS_tZtY' frameborder='0' allowfullscreen></iframe></div>

When using images as inputs, **fully-connected networks** (FCN) would have too many weights: slow learning and overfitting.

```{figure} ../img/fullyconnected.png
---
width: 50%
---
Fully-connected layers necessitate too many parameters on images.
```

**Convolutional layers** reduce the number of weights by **reusing** weights at different locations. This the principle of a convolution, which leads to fast and efficient learning.


```{figure} ../img/convolutional.png
---
width: 50%
---
Convolutional layers share weights on the image.
```

A **convolutional layer** extracts **features** of its inputs. $d$ filters are defined with very small sizes (3x3, 5x5...). Each filter is convoluted over the input image (or the previous layer) to create a **feature map**. The set of $d$ feature maps becomes a new 3D structure: a **tensor**. If the input image is 32x32x3, the resulting tensor will be 32x32xd. The convolutional layer has only very few parameters: each feature map has 3x3 values in the filter and a bias, i.e. 10 parameters. The convolution operation is **differentiable**: backprop will work.

```{figure} ../img/depthcol.jpeg
---
width: 50%
---
Convolutional layer. Source: <http://cs231n.github.io/convolutional-networks/>
```

The number of elements in a convolutional layer is still too high. We need to reduce the spatial dimension of a convolutional layer by **downsampling** it. For each feature, a **max-pooling** layer takes the maximum value of a feature for each subregion of the image (mostly 2x2). Pooling allows translation invariance: the same input pattern will be detected whatever its position in the input image. Max-pooling is also differentiable.

```{figure} ../img/maxpooling.jpg
---
width: 50%
---
Convolutional layer. Source: <http://cs231n.github.io/convolutional-networks/>
```

A **convolutional neural network** (CNN) is a cascade of convolution and pooling operations, extracting layer by layer increasingly  complex features. The spatial dimensions decrease after each pooling operation, but the number of extracted features increases after each convolution. One usually stops when the spatial dimensions are around 7x7. The last layers are fully connected. Can be used for regression and classification depending on the output layer and the loss function. Training a CNN uses backpropagation all along: the convolution and pooling operations are differentiable.

```{figure} ../img/lenet.png
---
width: 100%
---
Convolutional neural network.
```

The only thing we need to know is that CNNs are non-linear function approximators that work well with images.

$$\mathbf{y} = F_\theta (\mathbf{x})$$

The convolutional layers **extract complex features** from the images through learning. The last FC layers allow to approximate values (regression) or probability distributions (classification). 


## Autoencoders

<div class='embed-container'><iframe src='https://www.youtube.com/embed/IqrEdV9ejO8' frameborder='0' allowfullscreen></iframe></div>


The problem with FCN and CNN is that they **extract features** in supervised learning tasks: Need for a lot of annotated data (image, label). **Autoencoders** allows **unsupervised learning**, as they only need inputs (images). Their task is to **reconstruct** the input:

$$\mathbf{y} = \mathbf{\tilde{x}} \approx \mathbf{x}$$

The **reconstruction loss** is simply the **mse** between the input and its reconstruction.

$$
    \mathcal{L}_\text{autoencoder}(\theta) = \mathbb{E}_{\mathbf{x} \in \mathcal{D}} [ ||\mathbf{\tilde{x}} - \mathbf{x}||^2 ]
$$

Apart from the loss function, they are trained as regular NNs. Autoencoders consists of:

* the **encoder**: from the input $\mathbf{x}$ to the **latent space** $\mathbf{z}$.

* the **decoder**: from the latent space $\mathbf{z}$ to the reconstructed input $\mathbf{\tilde{x}}$.

```{figure} ../img/autoencoder-architecture.png
---
width: 100%
---
Autoencoder. Source: <https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html>
```

The **latent space** $\mathbf{z}$ is a **compressed representation** (bottleneck) of the inputs $\mathbf{x}$. It has to learn to compress efficiently the inputs without losing too much information, in order to reconstruct the inputs: Dimensionality reduction, unsupervised feature extraction.


In deep RL, we can construct the feature vector with an autoencoder. The autoencoder can be trained offline with a random agent or online with the current policy (auxiliary loss).


```{figure} ../img/autoencoder-RL.svg
---
width: 100%
---
Autoencoders can be used in RL to find a feature space where linear FA can apply easily.
```


## Recurrent neural networks

<div class='embed-container'><iframe src='https://www.youtube.com/embed/dkxIqBjldtY' frameborder='0' allowfullscreen></iframe></div>

FCN, CNN and AE are **feedforward neural networks**: they transform an input $\mathbf{x}$ into an output $\mathbf{y}$:

$$\mathbf{y} = F_\theta(\mathbf{x})$$

If you present a sequence of inputs $\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_t$ to a feedforward network, the outputs will be independent from each other:

$$\mathbf{y}_0 = F_\theta(\mathbf{x}_0)$$
$$\mathbf{y}_1 = F_\theta(\mathbf{x}_1)$$
$$\dots$$
$$\mathbf{y}_t = F_\theta(\mathbf{x}_t)$$

The output $\mathbf{y}_t$ does **not** depend on the history of inputs $\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{t-1}$. This not always what you want. If your inputs are frames of a video, the correct response at time $t$ might also depend on previous frames.
The task of the NN could be to explain what happens at each frame. As we saw, a single frame is often not enough to predict the future (**Markov property**).

```{figure} ../img/RNN-rolled.png
---
width: 30%
---
Recurrent neural network. Source: <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
```

A **recurrent neural network** (RNN) uses it previous output as an additional input (*context*). All vectors have a time index $t$ denoting the time at which this vector was computed. The input vector at time $t$ is $\mathbf{x}_t$, the output vector is $\mathbf{h}_t$:

$$
    \mathbf{h}_t = f(W_x \times \mathbf{x}_t + W_h \times \mathbf{h}_{t-1} + \mathbf{b})
$$


The input $\mathbf{x}_t$ and previous output $\mathbf{h}_{t-1}$ are multiplied by **learnable weights**:

* $W_x$ is the input weight matrix.
* $W_h$ is the recurrent weight matrix.

This is equivalent to a deep neural network taking the whole history $\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_t$ as inputs, but reusing weights between two time steps. The weights are trainable using **backpropagation through time** (BPTT). A RNN can learn the **temporal dependencies** between inputs.

```{figure} ../img/RNN-unrolled.png
---
width: 100%
---
Recurrent neural network, unrolled. Source: <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
```

A popular variant of RNN is **LSTM** (long short-term memory). In addition to the input $\mathbf{x}_t$ and output $\mathbf{h}_t$, it also has a **state** (or **memory** or **context**) $\mathbf{C}_t$ which is maintained over time. It also contains three multiplicative **gates**:

* The **input gate** controls which inputs should enter the memory.
* The **forget gate** controls which memory should be forgotten.
* The **output gate** controls which part of the memory should be used to produce the output.

```{figure} ../img/LSTM-cell2.png
---
width: 60%
---
LSTM cell. Source: <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
```

An obvious use case of RNNs in deep RL is for POMDP (partially observable MDP). If the individual states $s_t$ do not have the Markov property, the output of a LSTM does: The output of the RNN is a representation of the complete history $s_0, s_1, \ldots, s_t$. We can apply RL on the output of a RNN and solve POMDPs for free!

```{figure} ../img/capture-flag2.gif
---
width: 100%
---
LSTM layers help solving POMDP by concatenating and compressing the history. Source: <https://deepmind.com/blog/article/capture-the-flag-science>
```
