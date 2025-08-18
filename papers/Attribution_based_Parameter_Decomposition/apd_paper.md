# Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition

**Dan Braun*, Lucius Bushnaq*, Stefan Heimersheim*, Jake Mendel, Lee Sharkey†**  
*Core research contributor  
†Correspondence to lee@apolloresearch.ai  
Apollo Research‡  
‡Contributions statement below

---

## Abstract

Mechanistic interpretability aims to understand the internal mechanisms learned by neural networks. Despite recent progress toward this goal, it remains unclear how best to decompose neural network parameters into mechanistic components. We introduce *Attribution-based Parameter Decomposition* (APD), a method that directly decomposes a neural network's parameters into components that (i) are faithful to the parameters of the original network, (ii) require a minimal number of components to process any input, and (iii) are maximally simple. Our approach thus optimizes for a minimal length description of the network's mechanisms. We demonstrate APD's effectiveness by successfully identifying ground truth mechanisms in multiple toy experimental settings: Recovering features from superposition; separating compressed computations; and identifying cross-layer distributed representations. While challenges remain to scaling APD to non-toy models, our results suggest solutions to several open problems in mechanistic interpretability, including identifying minimal circuits in superposition, offering a conceptual foundation for `features', and providing an architecture-agnostic framework for neural network decomposition.

---

## Introduction

Mechanistic interpretability aims to improve the trustworthiness of increasingly capable AI systems by making it possible to understand their internals. The field's ultimate goal is to map the parameters of neural networks to human-understandable algorithms. A major barrier to achieving this goal is that it is unclear how best to decompose neural networks into the individual mechanisms that make up these algorithms, if such mechanisms exist [Bussmann et al. 2024, Sharkey et al. 2025]. This is because the mechanistic components of neural networks do not in general map neatly onto individual architectural components, such as individual neurons [Hinton 1981, Churchland & Sejnowski 2007, Nguyen et al. 2016], attention heads [Janiak et al. 2023, Jermyn et al. 2023], or layers [Yun et al. 2021, Lindsey et al. 2024, Meng et al. 2023].

Sparse dictionary learning is currently the most popular approach to tackling this problem [Lee et al. 2007, Yun et al. 2021, Sharkey et al. 2022, Cunningham et al. 2023, Bricken et al. 2023]. This method decomposes the neural activations of the model at different hidden layers into sets of sparsely activating latent directions. Then, the goal is to understand how these latent directions interact with the network's parameters to form circuits (or 'mechanisms') that compute the activations at subsequent layers [Cammarata et al. 2020, Olah et al. 2023, Sharkey et al. 2024, Olah et al. 2024]. However, sparse dictionary learning appears not to identify canonical units of analysis for interpretability [Bussmann et al. 2024]; suffers from significant reconstruction errors [Makelov et al. 2024, Gao et al. 2024]; optimizes for sparsity, which may not be a sound proxy for interpretability in the limit [Chanin et al. 2024, Till et al. 2024, Ayonrinde et al. 2024]; and leaves feature geometry unexplained [Engels et al. 2024, Mendel 2024], among a range of other issues (see [Sharkey et al. 2025] for review). These issues make it unclear how to use sparsely activating directions in activation space to identify the network's underlying mechanisms. Here, we investigate an approach to more directly decompose neural networks parameters into individual mechanisms.

There are many potential ways to decompose neural network parameters, but not all of them are equally desirable for mechanistic interpretability. For example, a neuron-by-neuron description of how a neural network transforms inputs to outputs is a perfectly accurate account of the network's behavior. But this description would be unnecessarily long and would use polysemantic components. This decomposition fails to carve the network at its joints because it does not reflect the network's deeper underlying mechanistic structure.

We therefore ask what properties an ideal mechanistic decomposition of a neural network's parameters should have. Motivated by the minimum description length principle, which states that the shortest description of the data is the best one, we identify three desirable properties:

- **Faithfulness**: The decomposition should identify a set of components that sum to the parameters of the original network.  
  *Faithfulness to the original network's parameters is subtly different from the 'behavioral faithfulness of a circuit', which has been studied in other literature [Wang et al. 2022]. Components that sum to the parameters of the original network will necessarily exhibit behavior that is faithful to the original network (assuming all components are included in the sum). But behavioral faithfulness does not imply faithfulness to a network's parameters, since different parameters may exhibit the same behavior. Our criterion is therefore stricter, and relates to a decomposition of parameters rather than [Wang et al. 2022]'s definition of a circuit.*
- **Minimality**: The decomposition should use as few components as possible to replicate the network's behavior on its training distribution.
- **Simplicity**: Components should each involve as little computational machinery as possible.

![Decomposing a target network's parameters into parameter components that are faithful, minimal, and simple.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/spd_target2-1.png)

Insofar as we can decompose a neural network's parameters into components that exhibit these properties, we think it would be justified to say that we have identified the network's underlying *mechanisms*: Faithfulness ensures that the decomposition reflects the parameters of and computations implemented by the network. Minimality ensures the decomposition comprises specialised components that play distinct roles. And simplicity encourages the components to be individual, basic computational units, rather than compositions of them.

To this end, we introduce *Attribution-based Parameter Decomposition (APD)*, a method that decomposes neural network parameters into components that are optimized for these three properties. In brief, APD involves decomposing the parameter vector of any neural network into a sum of *parameter components*. They are optimized such that they sum to the target parameters while only a minimal number of them are necessary for the causal process that computes the network's output for any given input. They are also optimized to be less complex individually than the entire network, in that they span as few directions in activation space as possible across all layers. APD can be understood as an instance of a broader class of *Linear Parameter Decomposition* (LPD) methods, a term we attempt to make precise in this paper.

Our approach leverages the idea that, for any given input, a neural network should not require all of its mechanisms simultaneously [Veit et al. 2016, Zhang et al. 2022, Dong et al. 2023]. On any given input, it should be possible to ablate unused mechanisms without influencing the network's computations. This would let us study the mechanisms in relative isolation, making them easier to understand.

For example, suppose a neural network uses only one mechanism to store the knowledge that '*The sky is blue*' in its parameters. Being a 'mechanism', as defined above, it is maximally simple, but may nevertheless be implemented using multiple neurons scattered over multiple layers of the model. Despite being spread throughout the network, we contend that there is a single vector in parameter space that implements this knowledge. On inputs where the model uses this stored knowledge, the model's parameters along this direction cannot be varied without changing the model's output. But on inputs where the model does not use this fact, ablating the model parameters along this direction to zero should not change the output.

Our method has several connections to other contemporary approaches in mechanistic interpretability, such as sparse dictionary learning [Sharkey et al. 2022, Cunningham et al. 2023, Bricken et al. 2023, Braun et al. 2024, Dunefsky et al. 2024, Ayonrinde et al. 2024, Lindsey et al. 2024], causal mediation analysis [Vig et al. 2020, Wang et al. 2022, Conmy et al. 2024, Syed et al. 2023, Kramár et al. 2024, Geiger et al. 2024], weight masking [Mozer & Smolensky 1988, Phillips et al. 2019, Csordás et al. 2021, de Cao et al. 2021], and others, while attempting to address many of their shortcomings. It also builds on work that explores the theory of computation in superposition [Vaintrob et al. 2024, Bushnaq & Mendel 2024].

This paper is structured as follows: We first describe our method in Section 2. In Section 3, we provide empirical support for our theoretical work by applying APD to three toy models where we have access to ground-truth mechanisms. First, in a toy model of superposition, APD recovers mechanisms corresponding to individual input features represented in superposition (Section 3.1). Second, in a model performing compressed computation -- where a model is tasked with computing more nonlinear functions than it has neurons -- APD finds parameter components that represent each individual function (Section 3.2). Third, when extending this model of compressed computation to multiple layers, APD is still able to learn components that represent the individual functions, even those that span multiple layers (Section 3.3). In Section 4, we discuss our results, the current state of APD, and possible next steps in its development, with conclusions in Section 5. We include a detailed discussion on related work in Section 4.1.

---

## Method: Attribution-based Parameter Decomposition

In this section, we outline our method, Attribution-based Parameter Decomposition (APD). First, we outline why we define 'mechanisms' as vectors in parameter space (Section 2.1). Then, we discuss how our method optimizes parameter components to be faithful, minimal, and simple, thus identifying the network's mechanisms (Section 2.2). While a brief description of APD suffices to understand our experiments, a more detailed description and motivation can be found in Appendix A.

### Defining 'mechanism space' as parameter space

To identify a neural network's mechanisms, we must first identify the space in which they live. The weights of neural networks can be flattened into one large parameter vector in *parameter space* (see Figure above). During learning, gradient descent iteratively etches a neural network's mechanisms into its parameter vector. This makes it natural to look for mechanisms in the same vector space as the whole network.

Vectors in parameter space also satisfy a broad range of criteria that we require individual mechanisms to have. *Mechanism space* should:

- **Span the same functional range as the target network**: We want mechanisms that perform a subcomponent of the algorithm implemented by the target neural network. We therefore expect mechanisms to lie somewhere in between "Doing everything the target network does" and "Doing nothing". Parameter space contains such mechanisms: The target network's parameter vector does everything that the target network does. And the zero parameter vector serves as a 'null mechanism'. Vectors that lie 'in between' serve as candidates for individual mechanisms.  
  *For this to be meaningful, we need a reasonable definition of what it means for vectors to lie 'in between' the target network's parameter vector and the zero vector. One reasonable definition is that vectors that are 'in between' should have a lower magnitude than the target network's parameter vector and have positive cosine similarity with the target network's parameters. This definition is implied by the method introduced in this work, although it does not optimize for these properties directly.*
- **Accommodate basis-unaligned mechanisms**: It has long been known that neural representations may span multiple neurons [Hinton 1981, Churchland & Sejnowski 2007, Nguyen et al. 2016]. However, even more recent work suggests that representations may span other architectural components, such as separate attention heads [Janiak et al. 2023, Jermyn et al. 2023] or even layers [Yun et al. 2021, Lindsey et al. 2024, Meng et al. 2023]. Vectors in parameter space span all of these components and can therefore implement computations that happen to be distributed across them.
- **Accommodate superposition**: Neural networks appear to be able to represent and perform computation on variables in superposition [Elhage et al. 2022, Vaintrob et al. 2024, Bushnaq & Mendel 2024]. We would like a space that can compute more functions than they have neurons. Vectors in parameter space support this requirement, theoretically [Bushnaq & Mendel 2024] and in practice, as we will demonstrate in our experiments.
- **Accommodate multidimensional mechanisms**: Some representations in neural networks appear to be multidimensional [Engels et al. 2024]. We therefore want to be able to identify mechanisms that can do multidimensional computations on these representations. Vectors in parameter space satisfy this requirement.

Having defined mechanism space as parameter space, we now want a method to identify a set of parameter components that correspond to the network's underlying mechanisms. In particular, we want to identify parameter components that satisfy the faithfulness, minimality, and simplicity criteria.

### Identifying networks' mechanisms using Attribution-based Parameter Decomposition

APD aims to minimize the description length of the mechanistic components used by the network *per data point* over the training dataset. It decomposes the network's parameters $\theta^* \in \mathbb{R}^N$ into a set of parameter components and directly optimizes them to be faithful, minimal, and simple. A discussion of how APD can be understood as an instance of a broader class of 'linear parameter decomposition' methods can be found in Appendix B, and more detailed discussion of how APD is based on the Minimum Description Length principle can be found in Appendix C.

**Optimizing for faithfulness:** We decompose a network's parameters $\theta^{*}_{l,i,j}$, where $l$ indexes the network's weight matrices and $i,j$ index rows and columns, by defining a set of $C$ parameter components $P_{c,l,i,j}$. Their sum is trained to minimize the mean squared error (MSE) with respect to the target network's parameters, $\mathcal{L}_{\text{faithfulness}}=\text{MSE}(\theta^*,\sum^C_{c=1} P_c)$.

**Optimizing for minimality:** The parameter components are also trained such that, for a given input, a minimal number of them is used to explain the network's output (see Figure below).

![Top: Step 1: Calculating parameter component attributions $A_c (x)$. Bottom: Step 2: Optimizing minimality loss $\mathcal{L}_{\text{minimality}}$.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/minimality_target-1.png)

1. **Attribution step:** We want to estimate the causal importance of each parameter component $P_c$ for the network's output on each datapoint $f_{\theta^*}(x)$. In this step, we therefore calculate the *attributions* of each parameter component with respect to the outputs, $A_c(x) \in \mathbb{R}$. It would be infeasibly expensive to compute this exactly, since it would involve a large number of causal interventions, requiring one forward pass for every possible combination of component ablations. We therefore use an approximation. In this work, we use gradient attributions [Mozer & Smolensky 1988, Molchanov et al. 2017, Nanda et al. 2022, Syed et al. 2023], but other attribution methods may also work. This step therefore involves one forward pass with the target model to calculate the output and one backward pass per output dimension to compute the attributions with respect to the parameters, which are used to calculate the attributions with respect to each parameter component.
2. **Minimality training step:** We sum only the top-$k$ most attributed parameter components, yielding a new parameter vector $\kappa(x) \in \mathbb{R}^N$, and use it to perform a forward pass. We train the output of the top-$k$ most attributed parameter components to match the target network's outputs by minimizing $\mathcal{L}_{\text{minimality}}=D(f_{\theta^*}(x), f_{\kappa(x)}(x))$, where $D$ is some distance or divergence measure. This step trains the *active* parameter components to better reconstruct the target network's behavior on a given data point. This should increase the attribution of active components on that data. In some cases, we also train some of the hidden activations to be similar on both forward passes, since it may otherwise be possible for APD to learn solutions that produce the same outputs using different computations.

In our experiments, we use batch top-$k$ [Bussmann et al. 2024] to select a fixed number of active parameter components for the minimality training step (a.k.a sparse forward pass) per batch. This sidesteps the issue of needing to select a specific number of active parameter components for each sample, although does present other issues.

**Optimizing for simplicity:** The components are also trained to be 'simpler' than the parameters of the target network. We would like to penalize parameter components that span more ranks or more layers than necessary by minimizing the sum of the ranks of all the matrices in active components: $\sum^C_{c=1} s_c(x)\sum_l\text{rank}(P_{c,l})$, where $s_c(x)\in\{0,1\}$ indicates active components. In practice, we minimize the $L_p$ norm of the singular values of weight matrices in active components using a loss $\mathcal{L}_{\text{simplicity}}(x)=\sum^C_{c=1} s_c(x)\sum_{l,m}\vert\vert \lambda_{c,l}\vert\vert^p_p$, where $\lambda_{c,l,m}$ are the singular values of parameter component $c$ in layer $l$. This is also known as the Schatten-$p$ norm.  
*Since $p\in(0,1)$, the Schatten-$p$ norm and $L_p$ norms here are technically quasi-norms. For brevity, we refer to them as norms throughout.*

**Biases:** Currently, we do not decompose the network's biases. Biases can be folded into the weights by treating them as an additional column in each weight matrix, meaning they can in theory be decomposed like any other type of parameter. However, in this work, for simplicity we treat them as their own parameter component that is active for every input, and leave their decomposition for future work.

**Summary:** In total, we use three losses:

1. A faithfulness loss ($\mathcal{L}_{\text{faithfulness}}$), which trains the sum of the parameter components to approximate the parameters of the target network.
2. A minimality loss ($\mathcal{L}_{\text{minimality}}$), which trains the top-$k$ most attributed parameter components on any given input to produce the same output (and some of the same hidden activations) as the target network, thereby increasing their attributions on those inputs.
3. A simplicity loss ($\mathcal{L}_{\text{simplicity}}$), which penalizes parameter components that span more ranks or more layers than necessary.

---

## Experiments: Decomposing neural networks into mechanisms using APD

In this section, we demonstrate that APD succeeds at finding faithful, minimal, and simple parameter components in three toy settings with known 'ground truth mechanisms'. These are:

1. [Elhage et al. 2022]'s toy model of superposition (Section 3.1);
2. A novel toy model of compressed computation, which is a model that computes more nonlinear functions than it has neurons (Section 3.2);
3. A novel toy model of cross-layer distributed representations (Section 3.3).

In all three cases, APD successfully identifies the ground truth mechanisms up to a small error. The target models are trained using AdamW [Loshchilov & Hutter 2019], though we also study a handcoded model in Appendix A. Additional figures and training logs can be found [here](https://api.wandb.ai/links/apollo-interp/j93iqupv). All experiments were run using [github.com/ApolloResearch/apd](https://github.com/ApolloResearch/apd). Training details and hyperparameters can be found in Appendix B.

### Toy Model of Superposition

Our first model is [Elhage et al. 2022]'s toy model of superposition (TMS), which can be written as:

$$
\hat{x}= \text{ReLU}(W^\top W x + b)
$$

with weight matrix $W \in \mathbb{R}^{m_1 \times m_2}$. The model is trained to reconstruct its inputs, which are sparse sums of one-hot $m_2$-dimensional input features, scaled to a random uniform distribution $[0,1]$. Typically, $m_1 < m_2$, so the model is forced to 'squeeze' representations through a $m_1$-dimensional bottleneck. When the model is trained on sufficiently sparse data distributions, it can learn to represent features in superposition in this bottleneck. For certain values of $m_1$ and $m_2$, the columns of the $W$ matrix often form regular polygons in the $m_1$-dimensional hidden activation space (see Figure below, leftmost panel).

![Results of running APD on TMS. Top row: Plot of the columns of the weight matrix of the target model, the sum of the APD parameter components, and each individual parameter component. Each parameter component corresponds to one mechanism, which in this model each correspond to one 'feature' in activation space [Elhage et al. 2022]. Bottom row: Depiction of the corresponding parametrized networks.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/tms_combined_diagram.png)

What are the 'ground truth mechanisms' in this toy model? Let us define a set of matrices $\{Z^{(c)}\}$ that are zero everywhere except in the $c^{\text{th}}$ column, where they take the values $W_{:, c}$:

$$
   Z^{(c)}_{:,j} = 
   \begin{cases} 
   W_{:,c} & \text{if } j = c, \\ 
   \textbf{0} & \text{otherwise}.
   \end{cases}
$$

The data are sparse, so only some of the model's weights $W$ are used on any given datapoint. Suppose we have a datapoint where only dataset feature $c$ is active. On this datapoint, we can replace $W$ with $Z^{(c)}$ and the model outputs would be almost identical (since the interference terms from inactive features should be small and below the learned $\text{ReLU}$ threshold). Intuitively, a column of $W$ is only 'used' if the corresponding data feature is active. This makes the matrices $\{Z^{(c)}\}$ good candidates for optimal 'minimality'. The matrices are also 'faithful', since $\sum_c Z^{(c)} = W$. They are also very simple because each matrix is rank 1, consisting of the outer product of the column of $W_{:,c}$ and the one-hot vector $e_c \in \mathbb{R}^{m_2}$ that indexes the nonzero column $c$:

$$
   Z^{(c)}=  W_{:,c} e_c^\top
$$

The matrices $\{Z^{(c)}\}$ are therefore reasonable candidates for the ground truth 'mechanisms' of this model. The $c^{\text{th}}$ 'mechanism' in this model technically corresponds to $Z^{(c)}$ and the $c^{\text{th}}$ element of the bias. For simplicity, we do not decompose biases in our current implementation and treat all biases as one component that is always active. We would therefore like APD to learn parameter components that correspond to them.

#### APD Results: Toy Model of Superposition

We find that APD can successfully learn parameter components $\{P_c\}$ that closely correspond to the matrices $\{Z^{(c)}\}$ (see Figure above). We observe that the sum of the components is equal to $W$ in the target network.

For illustrative purposes, we have focused on the setting with 5 input features ($m_2=5$) and a hidden dimension of 2 ($m_1=2$). However, training an APD model (and to a lesser extent, a target model) in this setting is very brittle and less effective than a higher-dimensional setting. Indeed, for the 2-dimensional hidden space setting, the results presented in this section required an adjustment of using attributions taken from the APD model rather than the target model (an adjustment that proved not to be beneficial in other settings). We expect that this brittleness has to do with the large amount of interference noise between the input features when projected onto the small 2-dimensional space. We thus also analyze a setting with 40 input features and 10 hidden dimensions. We use $\text{TMS}_{5-2}$ to denote the setting with 5 input features and 2 hidden dimensions, and $\text{TMS}_{40-10}$ to denote the setting with 40 input features and 10 hidden dimensions.

To show how close the learned parameter components are to the columns of $W$ in the target model, we measure the angle between each column of $W$ and the corresponding column in the component it lines up best with. We also measure how close their magnitudes are. To quantify the angles, we calculate the mean max cosine similarity (MMCS) [Sharkey et al. 2022]:

$$
\text{MMCS}(W, \{P_c\}) = \frac{1}{m_2}\sum_{j=1}^{m_2}\max_c\left(\frac{P_{c,:,j}\cdot W_{:,j}}{\| P_{c,:,j}\|_2 \| W_{:,j}\|_2}\right)
$$

where $c\in C$ are parameter component indices and $j\in[1,m_2]$ are input feature indices. A value of 1 for MMCS indicates that, for all input feature directions in the target model, there exists a parameter component whose corresponding column points in the same direction. To quantify how close their magnitudes are, we calculate the mean L2 Ratio (ML2R) between the Euclidean norm of the columns of $W$ and the Euclidean norm of the columns of the parameter components $P_c$ with which they have the highest cosine similarity:

$$
\text{ML2R}(W, \{P_c\}) = \frac{1}{m_2}\sum_{j=1}^{m_2} \frac{\| P_{\text{mcs}(j),:,j}\|_2}{\| W_{:,j}\|_2}
$$

where $\text{mcs}(j)$ is the index of the component that has maximum cosine similarity with weight column $j$ of the target model. A value close to 1 for the ML2R indicates that the magnitude of each parameter component is close to that of its corresponding target model column.

The MMCS and ML2R for both $\text{TMS}_{5-2}$ and $\text{TMS}_{40-10}$ are shown in the table below. We see in both settings that the MMCS values are approximately 1. This indicates that the parameter components are close representations of the target model geometrically. However, the ML2R is close to 0.9, implying there is some amount of 'shrinkage', reminiscent of feature shrinkage in SAEs [Jermyn et al. 2024, Wright et al. 2024]. We speculate that shrinkage in APD is caused by a forced trade-off between top-$k$ reconstruction $\mathcal{L}_{\text{minimal}}$ and the Schatten norm penalty $\mathcal{L}_{\text{simplicity}}$. In this specific case, we suspect it might be due to noise in the target model output due to high interference between the input features. The parameter components are incentivised by $\mathcal{L}_{\text{minimal}}$ to reconstruct this noise such that each learns small amounts of different ground truth mechanisms $\{Z^{(c)}\}$. Additional visualizations of the TMS APD models can be found in a [WandB report](https://api.wandb.ai/links/apollo-interp/j93iqupv).

| Setting         | MMCS         | ML2R         |
|----------------|-------------|-------------|
| $TMS_{5-2}$    | 0.998 ± 0.000 | 0.893 ± 0.004 |
| $TMS_{40-10}$  | 0.996 ± 0.003 | 0.935 ± 0.001 |

*Table: Mean max cosine similarity (MMCS) and mean L2 ratio (ML2R) with their standard deviations (to 3 decimal places) between learned parameter components and target model weights for TMS. The MMCS is very close to 1.0, indicating that every column in the target model has a corresponding column in one of the components that points in almost the same direction. The ML2R is below 1.0, indicating some amount of shrinkage in the components compared to the original model.*

![Decomposing TMS with APD.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/apd_tms_overview-1.png)

It is worth reflecting on the differences between the APD solution and the decompositions that other commonly used matrix decomposition methods would yield, such as singular value decomposition [Millidge et al. 2022, Meller et al. 2023] or non-negative matrix factorization [Petrov et al. 2021, Voss et al. 2021]. Those methods can find at most $\text{rank}(W)=\min(m_1, m_2)$ components, and therefore could not decompose $W$ into its ground truth mechanisms even in principle.

The toy model studied in this section was initially developed in order to demonstrate that neural networks can represent variables 'in superposition' using an overcomplete basis of the bottleneck activation space. However, our work decomposes the model, not activation space. Nevertheless, the mechanisms identified by our method *imply* an overcomplete basis in the activation space: The rank 1 mechanisms $Z^{(c)}$ can be expressed as an outer product of their (un-normed) left and right singular vectors $W_{:,c} e_c^\top$. The left singular vectors (corresponding to the columns of $W$) are an overcomplete basis of the $m_1$-dimensional hidden activation space. Parameter vectors can thus imply overcomplete bases for the activation spaces that they interact with, even though they do not form an overcomplete basis for parameter space.

The structure of this matrix decomposition is also revealing: We can think of $P_c$ as 'reading' from the $e_c^\top$ direction in the input space and projecting to the $W_{:,c}$ direction in the bottleneck activation space (see Figure above). Since we use $W$ and $W^\top$ in this model, in the next layer we can also think of this parameter component 'reading' from the $W_{:,c}^\top$ in the bottleneck activation space and projecting to the $e_c$ direction in the pre-ReLU activation space. In the backward pass, the roles are reversed: Directions that were 'reading' directions for activations become 'projecting' directions for gradients, and vice versa. In general, networks will learn parameter components consisting of matrices whose right singular vectors align with the hidden activations on the forward pass and whose left singular vectors align with gradients of the output with respect to the preactivations on the backward pass. Thus, they will ignore directions along which there are no activations, as well as directions that have no downstream causal effects.

---

### Toy Model of Compressed Computation

While the previous example (TMS) analyzed APD on a model that stored more features than dimensions, here we examine APD on a model performing more computations than it has neurons—a phenomenon that we term *compressed computation*. We chose this model because neural networks trained on realistic tasks may often perform more computations than they have neurons. Compressed computation is very similar to the "Computation in Superposition" toy model introduced by [Elhage et al. 2022], but our architecture and task differ. A key characteristic of representation in superposition [Elhage et al. 2022] and computation in superposition [Bushnaq & Mendel 2024] is a dependence on input sparsity. We suspect our model's solutions to this task might not depend on the sparsity of inputs as much as would be expected, potentially making 'compressed computation' and 'computation in superposition' subtly distinct phenomena. We leave a more detailed study of this distinction for future work.

We train a target network to approximate a function of sparsely activating inputs $x_i\in[-1, 1]$, using a Mean Squared Error (MSE) loss between the model output and the labels. The labels we train the model to predict are produced by the function $y_i = x_i + \text{ReLU}(x_i)$. Crucially, the task involves learning to compute more ReLU functions than the network has neurons.

The target network is a residual MLP, consisting of a residual stream width of $d_{\rm resid}=1000$, a single MLP layer of width $d_{\rm mlp}=50$, a fixed, random embedding matrix with unit norm rows $W_E$, an unembedding matrix $W_U=W_E^\top$, and $100$ input features. See the figure below for an illustration of the network architecture. The large residual stream $d_{\rm resid}=1000$ was chosen as the trained target network performed better than the naive monosemantic baseline in this setting (small values of $d_{\rm resid}$ lead to higher interference and thus worse model performance). We chose fixed, instead of trained, embedding matrices to make it simpler to calculate the optimal monosemantic baseline and to simplify training.

A naive solution to this task is to dedicate one neuron each to the computation of the first $d_{\rm mlp}$ functions, and to ignore the rest. This monosemantic baseline solution would perform perfectly for inputs that contained active features in only the first $d_{\rm mlp}$ input feature indices but poorly for all other inputs.

![The architecture of our Toy Model of Compressed Computation using a 1-layer residual MLP. We fix $W_E$ to be a randomly generated matrix with unit norm rows, and $W_U={W_E}^\top$.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid1_side-1.png)

To understand how each neuron participates in computing the output for a given input feature, we measure what we call the neuron's *contribution* to each input feature computation. For each neuron, this contribution is calculated by multiplying two terms:

1. How strongly the neuron reads from input feature $i$ (given by $W_\text{IN} {W_E}_{[:,i]}$).
2. How strongly the neuron's output influences the model's output for index $i$ (given by ${W_U}_{[i,:]} W_\text{OUT}$).

Mathematically, we compute neuron contributions for each input feature computation $i\in[0,99]$ by $({W_U}_{[i,:]} W_\text{OUT}) \odot (W_\text{IN} {W_E}_{[:,i]})$, where $\odot$ denotes element-wise multiplication. A large positive contribution indicates that the neuron plays an important role in computing the output for input feature $i$. The figure below (top) shows the neurons involved in the computation of the first $10$ input features of the target model and their corresponding contribution values.

The goal for APD in this setting is to learn parameter components that correspond to the computation of each input feature in the target model, despite these computations involving neurons that are used to compute multiple input features. For simplicity, we only decompose the MLP weights and do not decompose the target model's embedding matrix, unembedding matrix, or biases.

We found that parameter components often 'die' during training, such that no input from the training dataset can activate them. For this reason, we train with $130$ parameter components. This gives APD a better chance of learning all $100$ of the desired parameter components corresponding to unique input feature computations.

#### APD Results: Toy Model of Compressed Computation

Despite the target model computing more functions ($100$) than it has neurons ($50$), we find that APD can indeed learn parameter components that each implement $y_i = x_i + \text{ReLU}(x_i)$ for unique input dimensions $i\in \{0,\cdots,99 \}$. The figure below provides a visual representation of a set of learned parameter components. It shows how the computation that occurs for each input feature in the target network (top) is well replicated by individual parameter components in the APD model (bottom). We see that, for each input feature, there is a corresponding parameter component that uses the same neurons to compute the function as the target model does. Note that while we do not see a perfect match between the target model and the APD model, a perfect match would not actually be expected nor desirable: the neuron contribution scores of the target model can contain interference terms from the overlapping mechanisms of other features, which a single APD parameter component is likely to filter out. However, there is some 'shrinkage', similar to what we observe in the results on the TMS model. Here, much of the shrinkage is due to batch top-$k$ forcing APD on some batches to activate more components than there are features in the input, thereby spreading out input feature computations across multiple components.

![Similarity between target model weights and APD model components for the first 10 (out of 100) input feature dimensions. Top: Neuron contributions measured by $({W_U}_{[i,:]} W_\text{OUT}) \odot (W_\text{IN} {W_E}_{[:,i]})$ for each input feature index $i\in[0,9]$, where $\odot$ is an element-wise product.  Bottom: Neuron contributions for the predominant parameter components, measured by $\max_k [({W_U}_{[i,:]} {W_\text{OUT}}_k) \odot ({W_\text{IN}}_k {W_E}_{[:,i]})]$ for each feature index $i\in[0,9]$. The neurons are numbered from $0$ to $49$ based on their raw position in the MLP layer. An extended version of this figure showing all input features and parameter components can be found [here](https://api.wandb.ai/links/apollo-interp/h5ekyxm7).](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_weights_1layers_8qz1si1l.png)

Next, we investigate whether individual APD components have minimal influence on forward passes where their corresponding input feature is not active using a Causal Scrubbing-inspired experiment [Chan et al. 2022]: When performing a forward pass we ablate half of the APD model's parameter components, excluding the ones that correspond to the currently active inputs (the 'scrubbed' run). We compare this to ablating half of the parameter components *including* those that correspond to currently active inputs ('anti-scrubbed'). The figure below gives a visual illustration of the output of multiple 'scrubbed' and 'anti-scrubbed' runs for a one-hot input $x_{42} = 1$. We see that ablating unrelated components perturbs the output only slightly, barely affecting the overall shape.

![Output of multiple APD forward passes with one-hot input $x_{42} = 1$ over 10k samples, where half of the parameter components are ablated in each run. Purple lines show 'scrubbed' runs (parameter component corresponding to input index 42 is preserved), while green lines show 'anti-scrubbed' runs (component 42 is among those ablated). The target model output is shown in blue, which is almost identical to the output on the APD sparse forward pass (i.e. APD (top-$k$)). In this plot we only show the MLP output for clearer visualization. The embedding matrices are not decomposed and thus the residual stream contribution does not depend on APD components.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/feature_response_with_subnets_42_1layers_8qz1si1l.png)

To investigate whether this holds true for all components and on the training data distribution, we collect MSE losses into a histogram (see below). We find that the 'scrubbed' runs (i.e. ablating unrelated parameter components—pink histogram) does not cause a large increase in the MSE loss with respect to target network outputs. On the other hand, the anti-scrubbed runs (i.e. ablating parameter components that are deemed to be responsible for the computation—green histogram) does cause a large increase in MSE. This suggests that parameter components have mostly specialized to implement the computations for particular input features.  

However, some parameter components appear to partially represent secondary input feature computations. This causes the visibly bimodal distributions of the scrubbed runs that can be seen in the figure: When these components are ablated, the loss of the model may be high when the secondary input feature is active. These components have the opposite effect on the loss when they are not ablated in the anti-scrubbed runs, making both scrubbed and anti-scrubbed losses bimodal. Preliminary work suggests that this can be improved with better hyperparameter settings or with adjustments to the training process, such as using alternative loss functions or enforcing the APD components to be rank-1.

![MSE losses of the APD model on the sparse forward pass ("top-$k$") and the APD model when ablating half (50) of its parameter components ("scrubbed" when none of the components responsible for the active inputs are ablated and "anti-scrubbed" when they are ablated). The gray line indicates the loss for a model that uses one monosemantic neuron per input feature. The dashed colored lines are the mean MSE losses for each type of run.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_scrub_hist_1layers_8qz1si1l.png)

---

### Toy Model of Cross-Layer Distributed Representations

We have seen how APD can learn parameter components that represent computations on individual input features, even when those computations involve neurons that contribute to the computations of multiple input features. However, those computations take place in a single MLP layer. But realistic neural networks seem to exhibit cross-layer distributed representations [Yun et al. 2021, Lindsey et al. 2024]. In this section, we show how APD naturally generalizes to learn parameter components that represent computations that are distributed across multiple MLP layers.

We extend the model and task used in the previous section by adding an additional residual MLP layer (see figure below). This model still performs compressed computation, but now with representations that are distributed across multiple layers. In this model, $W_E$ is again a fixed, randomly generated embedding matrix with unit norm rows and $W_U=W_E^T$. We keep the residual stream width of $d_{\rm resid}=1000$ and $100$ input features, but our $50$ MLP neurons are now split across layers, with $25$ in each of the two MLPs. We train APD on this model with $200$ parameter components (allowing for $100$ to die during training).

![The architecture of our Toy model of Cross-Layer Distributed representations using a 2-layer residual MLP. We fix $W_E$ to be a randomly generated matrix with unit norm rows, and $W_U={W_E}^T$.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid2_side-1.png)

#### APD Results: Toy Model of Cross-Layer Distributed Representations

APD finds qualitatively similar results to the 1-layer toy model of compressed computation presented previously. We see that the APD model learns parameter components that use neurons with large contribution values in both MLP layers (see figure below, bottom). Again, we find that the computations occurring in each parameter component closely correspond to individual input feature computations in the target model (figure, top versus bottom). For confirmation that the target model and APD model in the 2-layer distributed computation setting yield results that closely match those observed in the 1-layer scenario, see Appendix B.

However, the results exhibit a larger number of imperfections compared to the 1-layer case. In particular, more components represent two input feature computations rather than one. As in the 1-layer case, we again notice that batch top-$k$ can cause some parameter components to not fully represent the computation of an input feature, and instead rely on activating multiple components for some input features.

![Similarity between target model weights and APD model components for the first 10 input feature dimensions in a 2-layer residual MLP. Top: Neuron contributions measured by $(W_E W_\text{IN}) \odot (W_\text{OUT} W_U)$ where $\odot$ is an element-wise product and $W_\text{IN}$ and $W_\text{OUT}$ are the MLP input and output matrices in each layer concatenated together.  Bottom: Neuron contributions for the learned parameter components, measured by $\max_k [({W_U}_{[i,:]} {W_\text{OUT}}_k) \odot ({W_\text{IN}}_k {W_E}_{[:,i]})]$ for each feature index $i\in[0,9]$. The neurons are numbered based on their raw position in the network, with neurons $0$ to $24$ in the first layer and neurons $25$ to $49$ in the second layer. An extended version of this figure showing all input features and parameter components can be found [here](https://api.wandb.ai/links/apollo-interp/h5ekyxm7).](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_weights_2layers_cb0ej7hj.png)

---

## Discussion

We propose APD, a method for directly decomposing neural network parameters into mechanistic components that are faithful, minimal, and simple. This takes a *parameters-first* approach to mechanistic interpretability. This contrasts with previous work that typically takes an *activations-first* approach, which decomposes networks into directions in activation space and then attempts to construct circuits (or 'mechanisms') using those directions as building blocks [Olah et al. 2020, Cammarata et al. 2020, Cunningham et al. 2023, Bricken et al. 2023, Marks et al. 2024].

A parameters-first approach has several benefits. The new lens it provides suggests straightforward solutions to many of the challenges presented by the activations-first approach to mechanistic interpretability. Nevertheless, it also brings novel challenges that will need to be overcome. In this section, we discuss both the potential solutions and challenges suggested by this new approach and suggest potential directions for future research.

### Addressing issues that seem challenging from an activations-first perspective

**The activations-first paradigm struggles to identify minimal circuits in superposition, while APD achieves this directly.**

APD optimizes a set of parameter components that are maximally simple while requiring as few as possible to explain the output activations of any given datapoint. It is possible to think about these parameter components as circuits, since they describe transformations between activation spaces that perform specific functional roles.

Identifying a method to obtain minimal circuits by building on sparse dictionary learning (SDL) -- which is an activations-first approach -- has proven difficult for several reasons. One reason is that even though SDL might identify sparsely activating latent directions, there is no reason to expect the connections between them to be sparse. This might result in dense interactions between latents in consecutive layers, which may be difficult to understand compared with latent directions that were optimized to interact sparsely. Another reason that SDL has struggled to identify concise descriptions of neural network parameters is the phenomenon of feature splitting [Bricken et al. 2023], where it is possible to identify an ever larger number of latents using ever larger sparse dictionaries. However, more latents means more connections between them, even if the transformation implemented by this layer is very simple! Descriptions of the connections may include an ever growing amount of redundant information. As a result, even very simple layers that transform latents in superposition may require very long description lengths.

**A parameters-first approach suggests a conceptual foundation for 'features'.**

A central object in the activations-first paradigm of mechanistic interpretability is a 'feature'. Despite being a central object, a precise definition remains elusive. Definitions that have been considered include [Elhage et al. 2022]:

1. 'Features as arbitrary functions', but this fails to distinguish between features that appear to be fundamental abstractions (e.g. a 'cat feature') and those that don't (e.g. a 'cat OR car' feature).
2. 'Features as interpretable properties', but this precludes features for concepts that humans don't yet understand.
3. 'Features as properties of the input which a sufficiently large network will reliably dedicate a neuron to representing'. This definition is somewhat circular, since it defines objects in neural networks using objects in other neural networks, and may not account for multidimensional features [Engels et al. 2024, Olah et al. 2024].

In our work, we decompose neural networks into parameter components that minimize mechanistic description length, which we call the network's 'mechanisms'. Note that a network's mechanisms are not equivalent to its 'features', but they might be related. Defining a network's features as 'properties of the input that activate particular mechanisms' seems to overcome the definitional issues above.

In particular, it overcomes the issues in Definition 1 because a set of maximally faithful, minimal, and simple components should learn to correspond to 'cat mechanisms' and 'car mechanisms', but not 'cat OR car mechanisms' (unless the target network actually did have specific machinery dedicated to 'cat OR car' computations). APD also does not rely on a notion of human interpretability, overcoming the issue with Definition 2. It also seems to overcome the issues of Definition 3, since the definition is not circular and should also be able to identify multidimensional mechanisms (and hence multidimensional features that activate them), although we leave this for future work.

The definition also overcomes issues caused by 'feature splitting', a phenomenon observed in SDL where larger dictionaries identify sets of different features depending on dictionary size, with larger dictionaries finding more sparsely activating, finer-grained features than smaller dictionaries. This happens because SDL methods can freely add more features to the dictionary to increase sparsity even if those features were not fundamental building blocks of computation used by the original network. APD components also need to be faithful to the target network's parameters when they are added together, meaning it cannot simply add more components in order to increase component activation sparsity or simplicity. To see this, consider a neural network that has a hidden layer that implements a $d$-dimensional linear map on language model data. A transcoder could learn ever more sparsely activating, ever more fine-grained latents to minimize its reconstruction and sparsity losses and represent this transformation. By contrast, the APD losses would be minimized by learning a single $d$-dimensional component that performs the linear map. The losses cannot be further reduced by adding more components, because that would prevent the components from summing up to the original network weights.

Incidentally, this thought experiment not only sheds light on feature splitting, but also sheds light on the difference between parameter components and 'features' as they are usually conceived. Parameter components are better thought of as "steps in the neural network's algorithm", rather than "representations of properties of the input". The network may nevertheless have learned mechanisms that specifically activate for particular properties of the input, which may be called 'features'.

**A parameters-first approach suggests an approach to better understanding 'feature geometry'.**

[Bussmann et al. 2024] showed that the Einstein SAE latent has a similar direction to other SAE latents that were German-related, physics-related, and famous people-related. This suggests that the latents that SDL identify lie on an underlying semantic manifold. Understanding what gives this manifold its structure should suggest more concise descriptions of neural networks. But SDL treats SDL latents as fundamental computational units that can be studied in isolation, thus ignoring this underlying semantic manifold [Mendel 2024]. We contend that the reason the Einstein latent points in the 'physics direction' (along with other physics-related latents) is because the network applies 'physics-related mechanisms' to activations along that direction. Therefore, by decomposing parameter space directly, we expect interpretability in parameter space to shed light on computational structure that gives rise to a network's SAE 'feature geometry'.

**Interpretability in parameter space suggests an architecture-agnostic method to resolving superposition.**

Neural network representations appear not to neatly map to individual architectural components such as individual neurons, attention heads, or layers. Representations often appear to be spread across various architectural components, as in attention head superposition [Jermyn et al. 2023] or cross layer distributed representations [Yun et al. 2021, Lindsey et al. 2024]. It is unclear how best to adapt SDL to each of these settings in order to tell concise mechanistic stories [Mathwin et al. 2024, Wynroe et al. 2024, Lindsey et al. 2024]. A more general approach that requires no adaptation would be preferred. Interpretability in parameter space suggests a way to overcome this problem in general, since any architecture can in theory be decomposed into directions in parameter space without the method requiring adaptation.

---

### Next steps: Where our work fits in an overall interpretability research and safety agenda

We had two main goals for this work. Our first goal was to resolve conceptual confusions arising in the activations-first, sparse dictionary learning-based paradigm of mechanistic interpretability. Our other main goal was to develop a method that builds on these conceptual foundations that can be applied to real-world models. However, APD is currently only appropriate for studying toy models because of its computational cost and hyperparameter sensitivity. We see two main paths forward:

1. **Path 1:** Develop APD-like methods that are more robust and scalable.
2. **Path 2:** Use the principles behind our approach to design more intrinsically decomposable architectures[^moe].

[^moe]: In particular, we are excited about research that explores how to pre-decompose models using mixtures-of-experts with many experts, where the experts may span multiple layers, like the parameter components in our work.

We are excited about pursuing both of these paths. In the rest of this section, we focus on Path 1, leaving Path 2 to future work.

We will outline what we see as the main challenges and exciting future research directions for building improved methods for identifying minimal mechanistic descriptions of neural networks in parameter space. To become more practical, APD must be improved in several ways. In particular, it should have a lower computational cost; be less sensitive to hyperparameters; and more accurate attributions. We may also need to fix outlying conceptual issues, such as the extent to which APD privileges layers. We also discuss several safety-oriented and scientific research directions that we think may become easier when taking a parameter-first approach to interpretability.

#### Improving computational cost

While developing this initial version of APD, we focused on conceptual progress over computational efficiency. At a glance, our method involves decomposing neural networks into parameter components that each have a memory cost similar to the target network. That would make it very expensive, scaling in the very worst case as something like $\mathcal{O}(N^2)$ where $N$ is the parameter count of the original model. However, there are several reasons to think this might not be as large an issue as it initially appears:

- **More efficient versions of APD are likely possible.** We think there exist paths toward versions of APD that are more computationally efficient. For the experiments in this paper, we often permitted the parameter components to be full rank. But theories of computation in superposition suggest that for a network to have many non-interfering components, they need to be low rank or localized to a small number of layers [Bushnaq & Mendel 2024]. A version of APD that constrains components to be lower rank and located in fewer layers would reduce their memory cost. Even if there are many high rank mechanisms, we think it may be possible to identify principles that let us stitch together many low rank components if the ground truth mechanisms are high rank, or let us use hierarchical representations of components. It may be possible to apply our method to models one layer at a time, like transcoders, which may save on memory costs of having to decompose every parameter at the same time.
- **Alternative approaches, such as SDL, may be even more expensive.** To use sparse dictionary learning to decompose a single layer's activation space may be relatively cheap compared with training an entire network. But even if it were possible to reverse engineer neural networks using sparse dictionaries (which is unclear), we would need to train sparse dictionaries on every layer in a network in order to reverse engineer it, which may be very expensive. It becomes even more expensive considering the need to identify or learn the connections between sparse dictionary latents in subsequent layers. At present, there is no reason to expect that it costs less to train sparse dictionaries on every layer than to perform APD. It may indeed cost much more to use SDL, since we do not know in advance what size of dictionary we need to use and how much feature splitting to permit. We suspect that a reasonably efficient version of APD, which aims to identify minimal mechanistic descriptions, to reverse engineer networks will fare favorably compared to using SDL to achieve similar feats, if SDL can be used for that purpose at all.
- **Our method suggests clear paths to achieving interpretability goals that other approaches have struggled to achieve.** Even if our method were more expensive than SDL-based approaches, our approach confers significant advantages (discussed above) that might make the computational cost worth it.

#### Improving robustness to hyperparameters

A practical issue at present is that the method is sensitive to hyperparameters. Extensive hyperparameter tuning was often required for APD to find the correct solution. Making the method more robust to hyperparameters is a high priority for future work. It is worth noting that we encountered fewer hyperparameter sensitivity issues when scaling up the method to larger toy models. It is likely that hyperparameter sensitivity was exacerbated due to the amount of interference noise between input feature computations in our experiments in small dimensions, and this may resolve itself when scaling up.

#### Improving attributions

One of the reasons that the method might not be robust is that the method currently uses gradient attributions, which are only a first order approximation of causal ablations [Mozer & Smolensky 1988, Molchanov et al. 2017]. Previous work, such as AtP [Nanda et al. 2022] and AtP* [Kramár et al. 2024] indicates that using gradients as first-order approximations to causal ablations work reasonably well, but become unreliable when gradients with respect to parameters become small due to e.g. a saturated softmax [Kramár et al. 2024]. This problem could potentially be alleviated by using integrated gradients [Sundararajan et al. 2017], learning masks for each parameter component on each datapoint during training [Caples et al. 2024], or other attribution methods instead.

#### Improving layer non-privileging

We want to find components in the structure of the learned network algorithm, rather than the network architecture. Thus, we would prefer our formalism to be entirely indifferent to changes of network parameterization that do not affect the underlying algorithm. However, APD is currently not indifferent to changes of network parameterization that mix network layers. Layers are therefore still slightly privileged by APD. This is because we optimize parameter components to be simple by penalizing them for being high rank, and the rank of weight matrices cannot be defined without reference to the network layers. Thus, if two components in different neural networks implement essentially the same computation, one in a single layer, the other in cross-layer superposition, the latter component may be assigned a higher rank. Therefore, while we think that APD can still find components that stretch over many layers, it may struggle to do so more than for components that stretch over fewer layers. We would need to find layer-invariant quantities that more accurately track the simplicity of components independent of their parametrization than effective rank. Speculatively, some variation of the weight-refined local learning coefficient [Wang et al. 2024] might fulfill this requirement.

#### Promising applications of interpretability in parameter space

If we can overcome these practical hurdles, we think that interpretability in parameter space may make achieving some of the safety goals of interpretability easier than with activations-first methods. For instance, if we have indeed identified a way to decompose neural networks into their underlying mechanisms, it will be readily possible to investigate mechanistic anomaly detection for monitoring purposes [Christiano et al. 2022]. Interpretability in parameter space may also be easier to perform precise model editing or unlearning of particular mechanisms [e.g. Meng et al. 2023], since model descriptions are given in terms of parameter vectors, which are the objects that we would directly modify.

We are also excited about applications of APD that might help answer important scientific questions. For instance, we suspect that APD can shed light on the mechanisms of memorizing vs. generalizing models [Henighan et al. 2023, Zhang et al. 2017, Arpit et al. 2017]; the mechanisms of noise robustness [Morcos et al. 2018]; or leveraging the fact that APD is architecture agnostic in order to explore potentially universal mechanistic structures that are learned independent of architecture [Li et al. 2015, Olah et al. 2020], such as convolutional networks, transformers, state space models, and more.

---

## Conclusion

This work introduces Attribution-based Parameter Decomposition (APD) as a fundamental shift in mechanistic interpretability: instead of analysing neural networks through their activations, we demonstrate that directly decomposing parameter space can reveal interpretable mechanisms that are faithful, minimal, and simple. Our approach suggests solutions to long-standing problems in mechanistic interpretability, including identifying minimal circuits in superposition, providing a conceptual foundation for 'features', enabling better understanding of 'feature geometry', and serving as an architecture-agnostic approach to neural network decomposition.

Our experiments demonstrate that APD can successfully identify ground truth mechanisms in multiple toy models: recovering features from superposition, separating compressed computations, and discovering cross-layer distributed representations.

Although our results are encouraging, several challenges remain before APD can be applied to real-world models. These include improving computational efficiency, increasing robustness to hyperparameters, and incorporating more robust attribution methods.

By decomposing neural networks into their constituent mechanisms, this work brings us closer to reverse engineering increasingly capable AI systems, helping to open the door toward a variety of exciting scientific and safety-oriented applications.

---

## Appendix

### More detailed description of the APD method

Suppose we have a trained neural network $f(x,\theta^*)$, mapping network inputs $x$ to network outputs $y=f(x,\theta^*)$, with parameters $\theta^*\in \mathbb{R}^N$.

We want to decompose into the 'mechanisms' that the network uses to compute its behavior. The network's parameters implement these mechanisms. We would therefore like some way to decompose a network's parameters into its constituent mechanisms.

We first define a set of parameter components

$$
P=\{P_1,\dots,P_C\}, \quad P_c \in \mathbb{R}^{N}, \quad \forall c \in \text{range}(1,\dots,C), \quad C\in \mathbb{N}.
$$

We want to train these parameter components such that they correspond to a network's mechanisms. We think that it is reasonable to define a network's mechanisms as a set of components that minimizes the total description length of the network's behavior, per data point, over the training dataset. In particular, we want to identify components that are maximally faithful to the target network, maximally simple, and where as few as possible are used to replicate the network's behavior on any given datapoint.

#### Linear Parameter Decomposition

We want the components to be faithful to the original model in the sense that ideally, they should sum to form the target model's parameters:

$$
\theta^*=\sum^C_{c=1} P_c
$$

However, we also want most of the parameter components $P_c\in \mathbb{R}^N$ to not be 'used' on any one network input $x$, in the sense that we can ablate all but a few of them without changing the outputs of the network.

**A definition of 'inactive' parameter components**

We could try to operationalize the idea of most components being 'inactive' in the sense of playing no meaningful role in the computation of the output by initializing a new network with a parameter vector $\kappa(x)$ composed of only a few of the most 'used' parameter components, and demanding that:

$$
f(x\vert \kappa(x))\approx f(x\vert \theta^*)
$$

where

$$
\kappa(x):=\sum^C_{c=1} s_c(x) P_c, \quad s_c(x) \in \{0,1\}, \quad \sum^C_{c=1} s_c(x)  \ll C.
$$

**A stricter definition of 'inactive' parameter components**

If the 'inactive' components are not playing any meaningful role in the computation of the output, we should also be able to ablate or partially ablate them in any combination, and still get the same result. In general, we should get the same network output for any parameter configuration along any monotonic 'ablation curve' $\gamma_c(x,t)$ where $t \in [0, 1]$:

$$
\gamma_c(x,0)= 1, \quad \gamma_c(x,1) = s_c(x), \quad \kappa(x,t):=\sum^C_{c=1} \gamma_c(x,t) P_c \\
f(x\vert \kappa(x,t))\approx f(x\vert\theta^*)
$$

This is a stricter definition of 'inactive' that seeks to exclude cases like components $P_c$ and $P_{c'}$ both being 'active' and important but canceling each other out. Without this stricter condition, we could have pathological solutions to the optimization problem. For example, suppose $P_1,\dots,\,P_{C-1}$ are a large set of parameter vectors for expert networks specialized to particular tasks that a target network $\theta^*$ is capable of. However, also suppose that these parameter vectors are completely unrelated to the underlying mechanisms in the target network $\theta^*$. Then, we could set the last component to

$$
P_{C}=\theta^*-\sum^{C-1}_{c=1} P_{c}
$$

The resulting components would add up to the target network, $\sum_c P_c=\theta^*$. And a single $P_c$ would always be sufficient to get the same performance as the target network. However, the components $P_c$ could be completely unrelated to the mechanistic structure of the target network. Requiring that the parameter components can be ablated part of the way and, in any combination, excludes counterexamples like this.

Equations above together define what we mean when we say that we want to decompose a network's parameter vector into a sum of other parameter vectors that correspond to distinct mechanisms of the network. The idea expressed in this definition is that different mechanisms combine *linearly* in parameter space to form the whole network. We call the class of methods that attempt to decompose neural network parameters into components that approximately satisfy these equations *Linear Parameter Decomposition* (LPD) methods.

**Component attributions**

Directly checking that the ablation curve condition is satisfied would be computationally intractable. Instead, for APD, we try to estimate whether the condition is approximately satisfied by calculating attributions of the output to each component. Currently, we do this using gradient attributions [Mozer & Smolensky 1988, Molchanov et al. 2017, Nanda et al. 2022, Syed et al. 2023], but other attribution methods may also work. This estimates the effect of ablating $P_c$ as:

$$
A_c(x):= \sqrt{\frac{1}{d_L} \sum^{d_L}_{o=1} {\left(\sum_{l,i,j}\frac{\partial f_o(x,\theta^*)}{\partial \theta_{l,i,j}}P_{c,l,i,j}\right)}^2}
$$

We take the average square of this term over all output indices $o$, where the final output layer has width $d_L$.

Previous work, such as [Nanda et al. 2022, Kramár et al. 2024], indicates that gradient-based first-order attribution methods can be somewhat accurate in many circumstances, but not always. For example, a saturated softmax in an attention head would render them inaccurate. Therefore, we expect that we might need to move to more sophisticated attribution methods in the future, such as integrated gradients [Sundararajan et al. 2017].

**The assumption of parameter linearity**

If neural networks do consist of a unique set of mechanisms in a meaningful sense, the ability of APD and any other LPD methods to recover that set of mechanisms relies on the assumption that the mechanisms are encoded in the network parameters in the linear manner specified above, at least up to some approximation. We call this the *assumption of parameter linearity*.

The assumption of parameter linearity approximately holds for all the neural networks we study in this paper. Figure (see Results) shows a test of the assumption on our compressed computation model, by checking whether inactive components in the APD decomposition can be ablated in random combinations without substantially affecting the end result.

Whether the assumption of parameter linearity is satisfied by all the non-toy neural networks that we might want to decompose is ultimately an empirical question. Current theoretical frameworks for computing arbitrary circuits in superposition [Vaintrob et al. 2024, Bushnaq & Mendel 2024] do seem to satisfy this assumption. They linearly superpose mechanisms in parameter space to perform more computations than the model has neurons[^superposition]. This provides a tentative theoretical basis to think that real models using computation in superposition do the same.

[^superposition]: In such a manner that the ablation curve equation should be satisfied up to terms scaling as ca. $\mathcal{O}(\epsilon)$, where $\epsilon$ is the noise level in the outputs of the target model due to superposition.

---

#### Deriving the losses used in APD from the Minimum Description Length Principle

**Idealised loss: Minimum description length loss, $\mathcal{L}_{\text{MDL}}$**

We suspect that many models of interest only use a fraction of the mechanisms they have learned to process any particular input. Thus, we want a decomposition of our models into components such that the total complexity of the components used on any given forward pass (measured in bits) is minimized.

**Motivating case: Parameter components that are rank 1 and localized in one layer.**

Suppose the elementary components in our model were all very simple, with each of them being implemented by a rank $1$ weight matrix $P_{c,l,i,j}= U_{c,l,i} V_{c,l,j}$ in some layer $l$ of the network[^rank1]. If we wanted to minimize the complexity used to describe the model's behavior on a given data point $x$, then we should minimize the number of components that have a non-zero causal influence on the output on that data point. In other words, we want to optimise the component attributions $A(x)$ to be sparse.

[^rank1]: We think that the total number of components $C$ here seems in theory capped to stay below the total number of network parameters $C=\mathcal{O}(N)$. See [Bushnaq & Mendel 2024] for discussion.

With a dense code, the attributions $A_c(x)$ on a given input $x$ would cost $\sum^C_{c=1} \alpha=C \alpha$ bits to specify, where $\alpha$ is the number of bits of precision we use for a single $A_c(x)$.

However, with a sparse code, we would instead need $\sum^C_{c=1} \vert\vert A_c(x)\vert\vert_0 \left(\alpha+\log_2(C)\right)$ bits, where $\vert\vert A_c(x)\vert\vert_0$ is the $L_0$ 'norm' of $A_c(x)$. If the parameter component attributions $A_c(x)$ are sparse enough, this can be a lot lower than $C\alpha$. This leverages the fact that we can list only the indices and attributions of the subnets with non-zero $A_c(x)$. This requires $\log_2(C)$ bits for the index and $\alpha$ bits for the attribution.

**General case: Parameter components that have arbitrary rank and may be distributed across layers.**

We do not expect all the parameter components of models to always be rank $1$ matrices; they may be arbitrary rank and span multiple layers[^lowrank]. We can treat this similar to the motivating case above, but where a parameter component that consists of a rank $2$ matrix can be represented as two rank $1$ matrices that always co-activate.

[^lowrank]: Nevertheless, current hypotheses for how models might implement computations in superposition suggest that components would tend to be low-rank. [Bushnaq & Mendel 2024]. Otherwise, there would just not be enough spare description length to fit all the (high rank) parameter components that are necessary to do the computation in superposition into the network.

If two rank $1$ matrices almost always coactivate, then we can describe their attributions in two ways:

1. If we consider them as **two separate components**, then we would need $2\log_2(C)+2\alpha$ bits to describe their attributions for each data point they activate on ($\log_2(C)$ for the index and $2\alpha$ for the two attributions).
2. However, if we consider them as **one separate component**, then we only need one index to identify both of them, and therefore only need $\log_2(C)+2\alpha$ bits

This means that we may be able to achieve shorter description lengths using a mixed coding scheme that allows for both dense and sparse codes. Thus, if we use a mixed coding scheme that allows rank $1$ parameter components to be aggregated into higher dimensional components, it gives us a description length of

$$
\mathcal{L}_{\text{MDL}}(x) = \sum^C_{c=1} \vert\vert A_c(x)\vert\vert_0 \left(\alpha \sum_l \text{rank}(P_{c,l})+\log_2(C)\right) \\
= \log_2(C) \left(  \sum^C_{c=1} \vert\vert A_c(x)\vert\vert_0 \right) + 
\alpha \left(  \sum^C_{c=1} \vert\vert A_c(x)\vert\vert_0  \sum_l \text{rank}(P_{c,l}) \right) \\
=: \log_2(C) \mathcal{L}_{\text{minimality}}^{\text{idealized}}(x) + \alpha \mathcal{L}_{\text{simplicity}}^{\text{idealized}}(x)
$$

where $\sum_l\text{rank}(P_{c,l})$ is the total rank of component $P$ summed over the weight matrices in all the components of the network.

Optimizing our components $P_c$ to minimize $\mathcal{L}_{\text{MDL}}(x)$ would then yield a decomposition of the network that uses only small values for the total number of active components and the total rank of the active components on a particular forward pass.

The prefactor $\alpha$ in this equation then sets the point at which two lower-rank components coactivate frequently enough that merging them into a single higher-rank component lowers the overall loss. Thus, $\alpha$ is effectively a hyperparameter controlling the resolution of our decomposition. As $\alpha$ increases, the threshold for merging components rises, with all components becoming rank $1$ in the limit $\alpha\rightarrow \infty$. If we set $\alpha=0$, all components would merge, so our decomposition would simply return the target network's parameter vector.

**Full idealised MDL loss**

For our the loss term that we use to train our parameter components, we want a decomposition that approximately sums to the target parameters and minimises description length. We can accomplish this by adding a faithfulness loss

$$
\mathcal{L}_{\text{faithfulness}} = \sum_{l,i,j}{\left( \theta^{*}_{l,i,j}- \sum^C_{c=1} P_{c,l,i,j}\right)}^2
$$

to our minimum description length loss. Our full loss is then[^logc]:

$$
\mathcal{L}_{\text{faithfulness}}+\mathcal{L}_{\text{MDL}}(x) = \mathcal{L}_{\text{faithfulness}}+\beta \mathcal{L}_{\text{minimality}}^{\text{idealized}}(x)+\alpha \mathcal{L}_{\text{simplicity}}^{\text{idealized}}(x)
$$

[^logc]: Here, we've absorbed $\log(C)$ from $\mathcal{L}_{\text{MDL}}$ in the previous section into $\beta$.

However, this idealized loss would be difficult to optimize since the $L_0$ 'norm' $\vert\vert A_c(x)\vert\vert_0$ and $\text{rank}(P_{c})$ are both non-differentiable. We therefore must optimize a differentiable proxy of this loss instead.

We have devised two different proxy losses for this, leading to two different implementations of APD. The first uses a **top-$k$ formulation**, whereas the second assigns an $L_p$ penalty to attributions. We primarily use the top-$k$ formulation in our work. But we include the $L_p$ version for explanatory purposes.

#### Practical loss: Top-$k$ formulation of APD

**Approximating $\mathcal{L}_{\text{minimality}}^{\text{idealized}}$ in the top-$k$ formulation**

We can approximate optimizing for the loss above with a top-$k$ approach: We run the network once on data point $x$ and collect attributions $A_c(x)$ for each parameter component $P_c$. Then, we select the parameter components with the top-$k$ largest attributions and perform a forward pass using only those components.

$$
s_c(x) \in \{0,1\}\,\\
s_c(x)=\text{top-k}(\{A_c(x)\})\\
\kappa(x):= \sum^C_{c=1} s_c(x) P_c
$$

This 'sparse' forward pass should ideally only involve the structure in the network that is actually used on this specific input, so it should give the same result as a forward pass using all $P_c$. We can optimise for this using a loss

$$
\mathcal{L}_{\text{minimality}}(P\vert \theta^*, X) = D\left(f(x\vert \theta^*),f(x\vert \kappa(x))\right)
$$

where $D$ is some distance measure between network outputs, e.g. MSE loss or KL-divergence. Minimising $\mathcal{L}_{\text{minimality}}$ for a small $k$ then approximately minimises $\sum^C_{c=1} \vert\vert A_c(x)\vert\vert_0$ in the ideal loss.

**Reconstructing hidden activations**

It is possible that reconstructing the network outputs on the sparse forward pass is not a strong enough condition to ensure that the components we find correspond to the mechanisms of the network, particularly since our attributions are imperfect. To alleviate this, we can additionally require some of the model's hidden activations on the sparse forward pass to reconstruct the target model's hidden activations. This can also aid training dynamics in deeper models, as APD can match the target model layer by layer instead of needing to re-learn everything from scratch. However, theories of computation in superposition predict that unused components still contribute noise to the model's hidden preactivations before non-linearities, which is then filtered out [Hänni et al. 2024, Bushnaq & Mendel 2024]. So we do not necessarily want to match the hidden activations of the target model everywhere in the network. Finding a principled balance in this case is still an open problem. We use a hidden activation reconstruction loss for our single and multilayer models of compressed computation.

**Approximating $\mathcal{L}_{\text{simplicity}}^{\text{idealized}}$ in the top-$k$ formulation**

To approximate $\mathcal{L}_{\text{simplicity}}^{\text{idealized}}$, we need some tractable objective function that approximately minimises $\text{rank}(P_{c})$. We use the *Schatten norm*: The rank of a matrix $M$ can be approximately minimised by minimising $\vert\vert M\vert\vert_p$ [Giampouras et al. 2020] with $p\in(0,1)$:

$$
\vert\vert M\vert\vert_p:=\left(\sum_m \vert\lambda_m \vert^p\right)^{\frac{1}{p}}
$$

Here, $\lambda_m$ are the singular values of $M$. So, we can approximate $\text{rank}(P_c)$ in the loss with

$$
\sum^C_{c=1}\vert\vert P_c\vert\vert^p_p =\sum^C_{c=1}\sum_{l,m}\vert\lambda_{c,l,m} \vert^p
$$

where $\lambda_{c,l,m}$ is singular value $m$ of component $c$ in layer $l$.

Performing a singular value decomposition for every component at every layer every update step would be cumbersome. We can circumvent this by parametrizing our components in factorized form, as a sum of outer products of vectors $U,V$:

$$
P_{c,l,i,j}:= \sum_k U_{c,l,m,i} V_{c,l,m,j}
$$

If we now replace $\lambda_{c,l,m}$ with

$$
\lambda_{c,l,m}\rightarrow \left(\sum_{i,j} U^2_{c,l,m,i} V^2_{c,l,m,j}\right)^{\frac{1}{2}}
$$

then $V_c$ and $U_c$ will be incentivised to effectively become proportional to the right and left singular vectors for subnet $P_c$.

The Schatten norm of $P_c$ can then be written in factorized form as:

$$
\mathcal{L}_{\text{simplicity}}(x) = \sum^C_{c=1} s_c(x)\sum_{l,m}\left(\sum_{i,j} U^2_{c,l,m,i} V^2_{c,l,m,j}\right)^{\frac{p}{2}}
$$

**Full set of loss functions in the top-$k$ formulation**

To summarise, our full loss function is

$$
\mathcal{L}(x) = \mathcal{L}_{\text{faithfulness}}+\beta \mathcal{L}_{\text{minimality}}(x)+\alpha \mathcal{L}_{\text{simplicity}}(x)\\
\mathcal{L}_{\text{faithfulness}} = \sum_{l,i,j}{\left( \theta^{*}_{l,i,j}- \sum^C_{c=1} P_{c,l,i,j}\right)}^2\\
\mathcal{L}_{\text{minimality}}(x) = D\left(f(x\vert \theta^*),f(x\vert \sum^C_{c=1} s_c(x) P_{c})\right)\\
\mathcal{L}_{\text{simplicity}}(x) = \sum^C_{c=1} s_c(x)\sum_{l}\vert\vert P_c\vert\vert^p_p
$$

The components $P_c$ are parametrised as

$$
P_{c,l,i,j}:= \sum_m U_{c,l,m,i} V_{c,l,m,j}
$$

The top-$k$ coefficients $s_c(x)$ are chosen as

$$
s_c(x) \in \{0,1\}\,\\
s_c(x)=\text{top-k}(\{A_c(x)\})
$$

where $A_c(x)$ are attributions quantifying the effect of components $P_c$ on the network, computed with attribution patching as above, or with some other attribution method. Finally, $\vert\vert P_c\vert\vert_p$ denotes the Schatten norm, and $p\leq 1.0$ is a hyperparameter.

$\mathcal{L}_{\text{minimality}}(x)$ may include additional terms penalizing the distance $D$ between some of the hidden activations of the target model $\theta^*$, and the sparse forward pass using parameters $\sum^C_{c=1} s_c(x) P_{c}$.

We use batch top-$k$ instead of top-$k$ [Bussmann et al. 2024], picking the components with the largest attributions over a batch of datapoints instead of single inputs.

#### Alternative practical loss: APD formulation that uses an $L_p$ penalty on attributions

As an alternative to the top-$k$ loss, we can also approximately optimize for the idealized loss with an $L_p$ approach. Optimizing the $L_p$ norm with $p\leq 1$ will tend to yield solutions with small $L_0$ 'norm', while still being differentiable. So we can replace $\vert\vert A_c(x)\vert\vert_0$ in the loss with $\vert A_c(x)\vert^p$. Our losses would then be

$$
\mathcal{L}_{\text{minimality}}^{L_p}(x) = \sum^C _{c=1} \vert A_c(x)\vert^{p_1} \\
\mathcal{L}_{\text{simplicity}}^{L_p}(x) = \sum^C_{c=1}\sum_l\vert A_c(x)\vert^{p_1} \,\left(\sum_{i,j} U^2_{c,l,m,i} V^2_{c,l,m,j}\right)^{\frac{p_2}{2}}
$$

where $p_1,p_2\leq 1.0$ are the $p$-norms of the attributions and the Schatten norm of the matrices respectively.

![Optimizing $\mathcal{L}_{\text{minimality}}^{L_p}$](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/l_p_formulation-1.png)

We did not thoroughly explore this implementation because our early explorations that used the $L_p$ approach did not work as well as our top-$k$ implementation for unknown reasons. We may revisit this approach in the future.

---

### Further experiments

#### Hand-coded gated function model: Another cross-layer distributed representation setting

![Hand-coded gated function model: The four functions $f_i(x)$ implemented by the hand-coded gated function model (solid lines), and the outputs of the top-$k$ forward pass of the APD-decomposed model (dashed lines). The APD model almost perfectly matches the hand-coded network.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/model_functions_paper.png)

**Setup**

In this task, we hand-code a target network to give an approximation to the sum of a set of trigonometric functions, governed by a set of control bits. The functions being approximated are of the form $F_i(x)= a_i \cos(b_i x + c_i) + d_i \sin(e_i x + f_i)+h_i$ with randomly generated coefficients $\{a_i,b_i,c_i,d_i,e_i,f_i,g_i\}$ drawn from uniform distributions (see table below) for each unique function $i$.

The input to the network is a vector, $(x, \alpha_0, \cdots, \alpha_{n-1})$, whose entries are a scalar $x\in[0, 5]$ and a set of $n$ binary control bits $\alpha_i\in\{0,1\}$. The control bits $\alpha_i$ are sparse, taking a value of $1$ with probability $p=0.05$ and $0$ otherwise. A function is only "active" (i.e. it should be summed in the output of the network) when its corresponding control bit is on.

Similar to our model of cross-layer distributed representations, we use 2-layer residual MLP network with ReLU activations. This model is hand-crafted to have $n$ clearly separable mechanisms that each approximate a unique trigonometric function. Notably, each function is computed by a unique set of neurons.

The output of the target model is a piecewise approximation of $y(x) = \sum_i \alpha_i F_i(x)$ with $n$ functions $y_i(x)$.

| **Coefficient** | **Range** |
|----------------|-----------|
| $a$ | $\mathcal{U}(-1, 1)$ |
| $b$ | $\mathcal{U}(0.1, 1)$ |
| $c$ | $\mathcal{U}(-\pi, \pi)$ |
| $d$ | $\mathcal{U}(-1, 1)$ |
| $e$ | $\mathcal{U}(0.1, 1)$ |
| $f$ | $\mathcal{U}(-\pi, \pi)$ |
| $g$ | $\mathcal{U}(-1, 1)$ |

*Table: Ranges of coefficients sampled from uniform distributions for the functions used in the hand-coded gated function model.*

In our experiments, we use a total of $n=4$ unique functions, with each function using $m=10$ neurons to piecewise-approximate the functions $F_i(x)$. We show these approximated functions in the figure above (solid lines). The 5 inputs of our network ($x$, $\alpha_0$, $\alpha_1$, $\alpha_2$, $\alpha_3$) are stored in the first 5 dimensions of the residual stream, alongside a dimension that we read off as the output of the network ($\hat{y}(x)$). To hand-code the piecewise approximation of the individual functions $y_i$ we randomly select $m$ neurons from the MLPs, typically distributed across layers. This also means that the value of $\hat{y}_i$ is not represented in the intermediate layers, but only in the final layer.

We show the weights of the hand-coded target network in the leftmost panel of the figure below. The graph shows the residual MLP network, with weights shown as lines. Each neuron is monosemantic, that is, it is used to approximate one of the $F_i(x)$ functions. Each neuron connects to the respective control bit $\alpha_i$ as well as the $x$ input. All neurons write to the output activation, which is the last dimension in the residual stream. The line color in the figure indicates which task (i.e. which function $F_i$) the weight implements; the line width indicates the magnitude of the weight.

When applied to this network, APD should partition the network weights $\theta^*$ into $C=n$ parameter components $P_c$, each corresponding to the weights for one approximated function $F_i(x)$ (i.e. of one colour).

**Results**

We find that APD can decompose this network into approximately correct parameter components. However, APD is particularly difficult to train in this setting, with only minor changes in hyperparameters causing large divergences. We hypothesize that this may be due to the fact that the ground truth network is itself hand-coded, not trained. We show a cherry-picked example (out of many runs that vary the number of MLP layers and number of functions) in the figure below.

![The parameters of the hand-coded gated function model decomposed into parameter components. Leftmost panel: The hand-coded network parameters, colored by the unique functions $F_i(x)$. Other panels: The parameter components identified by APD, coloured by the function they correspond to in the target model, or purple if the weight is zero in the target model.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/subnetworks_graph_plots.png)

The figure above shows the target network weights (leftmost column), and their decomposition into the four APD-generated components (remaining columns). We color the weights by which feature they correspond to in the target model, or purple if the weight is not present in the target model. We observe that the components mostly capture one function each (most weights within a parameter component are the same color).

However, the solution is not perfect. Some weights that are not present in the target network are nevertheless nonzero in some of the parameter components. Additionally, the $W_{\rm out}^1$ weights of parameter component $2$ and $W_{\rm out}^0$ weights of parameter component $3$ seem to be absorbed into other parameter components. This may be due to the difficulty in training APD on a handcoded model as mentioned earlier, or may be a symptom of the simplicity loss $\mathcal{L}_{\text{simplicity}}$ not being fully layer-independent, causing an over-penalization of weights being in a layer on their own.

---

### Further analyses

#### Analysis of the compressed computation target model

In this section we provide more details about the performance of the target residual MLP model that is used to train APD, as discussed in the Toy Model of Compressed Computation section.

Recall that we train the target network to approximate $y_i = x_i + \text{ReLU}(x_i)$. Note that the model output can be written as

$$
\mathbf{y} = W_U W_E \mathbf{x} + W_U W_{\rm out} \text{ReLU}(W_{\rm in} W_E \mathbf{x})
$$

Since $W_E$ consists of random unit vectors and is not trained. Also, $W_U = W_E^T$. As a result, the first summand already approximates a noisy identity and the second summand mostly approximates the ReLU function.

The figure below (left) shows the output of the model for an arbitrary one-hot input ($x_{42}=1$). We see that the output $\hat{x}_{42}\approx 1.6$ is close to the target value of $2.0$, and the remaining outputs $\hat{x}_{i\neq 42}$ are close to $1.0$. We checked whether the noise in the $\hat{x}_{i\neq 42}$ outputs comes from the $W_U W_E$ or MLP term, and found that it is dominated by the MLP term[^noise].

[^noise]: This is not the case for small embedding sizes, such as $d_{\rm resid}=100$. This is why we chose a large embedding size to focus on the MLP noise.

We confirm that $\hat{x}_{42}$ indeed approximates a ReLU function for $\hat{x}\in[-1,1]$ in the figure below (right panel), though not perfectly. It appears to systematically undershoot the labels. We expect that this is due to the MSE loss: Although the model could scale the outputs (by scaling e.g. $W_{\rm out}$) to match $y_{42} = 2.0$, it would also increase the loss overall.

So far we have focused on the arbitrary input index $42$. The second figure below repeats the same experiment but overlaying the results of all $100$ input features (lines color indicating the input feature index). We can see that the model treats all input features approximately the same.

![Output of the $1$-layer residual MLP target model compared to true labels for a single active input. Left: Output at all output indices for single one-hot input $x_{42}=1$. Right: Output at index $j=42$ for inputs with $x_{42}\in[0,1]$ and $x_j=0$ for $j\neq 42$.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_feature_response_single_1layers.png)

![Output of the $1$-layer residual MLP target model compared to true labels for the full set of $100$ one-hot inputs. Left: Output at all output indices over the set of inputs. The point color indicates the active input feature, and label values are in red. Right: Output at index $i$ for inputs with $x_{i}\in[0,1]$ and $x_j=0$ for $j\neq i$. Line colors indicate the input feature index.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_feature_response_multi_1layers.png)

#### Analysis of the compressed computation APD model

For a setting like the compressed computation task, where the dataset consists of input features activating independently with probability $p$, a natural choice for the batch top-$k$ hyperparameter is a value close to $p$ multiplied by the number of input features. In our experiments, this would be $0.01\times100=1$. For this value of batch top-$k$ (and similar), there are batches in which APD must activate more parameter components than there are active features, and likewise, batches in which APD must activate fewer parameter components than there are active features. In our $1$-layer and $2$-layer residual MLP experiments, we chose the value of batch top-$k=1.28$ to be such that in almost no batches would there be more active input features than active components (we use a batch size of $256$).

The benefits of choosing this large batch top-$k$ value are:

1. APD can learn to handle rarer samples with many active input features.
2. Since there are very rarely more active input features than active components, the components are not encouraged to represent the computations of multiple input features.

However, since there are extra active parameter components in most batches, APD exhibits a behavior where, for a subset of input features, it represents part of its computation in multiple parameter components. This phenomenon is illustrated in the figure below, where the APD model achieves a low loss across all input features when using its trained batch top-$k=1.28$ setting (bottom). However, when constrained to activate only a single parameter component per sample, the model exhibits large losses for a non-negligible subset of the input features (top). These results are based on samples where only one input feature is active. This behavior is further characterized in the second figure below. Samples with higher MSE loss under single-component activation tend to require more parameter components on the training distribution with batch top-$k=1.28$.

![MSE for APD trained with batch top-$k=1.28$ in the $1$-layer residual MLP setting for samples with a single active input feature (i.e. one-hot), averaged over $100$k samples. Top: Comparison of the target model with the APD model when activating exactly one parameter component in each sample (i.e. top-$k=1$). Bottom: Comparison of the target model with the APD model using batch top-$k=1.28$. The batch top-$k$ mask is applied to the original training distribution and then samples without exactly one active input feature are filtered out.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_per_feature_performance_1layers_8qz1si1l.png)

![MSE for APD trained with batch top-$k=1$ in the $1$-layer residual MLP setting for samples with a single active input feature (i.e. one-hot), averaged over $100$k samples. Top: Comparison of the target model with the APD model when activating exactly one parameter component in each sample (i.e. top-$k=1$). Bottom: Comparison of the target model with the APD model using batch top-$k=1$. The batch top-$k$ mask is applied to the original training distribution and then samples without exactly one active input feature are filtered out.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_per_feature_performance_1layers_9a639c6w.png)

![Relationship in the $1$-layer residual MLP setting between: (y-axis) the average number of active APD parameter components when using batch top-$k=1.28$, and (x-axis) the MSE between the target model outputs and the APD model when activating exactly one parameter component in each sample (i.e. top-$k=1$). MSE is measured only on samples with a single active input feature.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_avg_components_scatter_1layers_8qz1si1l.png)

As shown in the second figure above, we see that training with a reduced batch top-$k$ value of $1$ (rather than $1.28$) reduces the number of input features that have a large MSE loss when only activating a single parameter component. However, the downside of using a smaller top-$k$ value is that we end up with more components that fully represent two different input feature computations, rather than one. This should not be surprising; when top-$k$ is smaller, there are more batches in which the number of active input features is larger than the number of active components. APD is then incentivized to represent multiple input feature computations in a single parameter component to achieve a smaller $\mathcal{L}_{\text{minimality}}$ (though, at the cost of a larger $\mathcal{L}_{\text{simplicity}}$).

#### Analysis of the cross-layer distributed representations target model

In the figures below, we show that the trained target model for the cross-layer distributed representations setting (i.e. $2$-layer residual MLP) is qualitatively similar to the target model in the compressed computation setting (i.e. $1$-layer residual MLP) we analyzed above.

![Output of the $2$-layer residual MLP target model compared to true labels for a single active input. Left: Output at all output indices for single one-hot input $x_{42}=1$. Right: Output at index $j=42$ for inputs with $x_{42}\in[0,1]$ and $x_j=0$ for $j\neq 42$.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_feature_response_single_2layers.png)

![Output of the $2$-layer residual MLP target model compared to true labels for the full set of $100$ one-hot inputs. Left: Output at all output indices over the set of inputs. The point color indicates the active input feature, and label values are in red. Right: Output at index $i$ for inputs with $x_{i}\in[0,1]$ and $x_j=0$ for $j\neq i$. Line colors indicate the input feature index.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_feature_response_multi_2layers.png)

#### Analysis of the cross-layer distributed representations APD model

Here, we show that the APD model for the cross-layer distributed representations setting (i.e. $2$-layer residual MLP) is qualitatively similar to the APD model in the compressed computation setting (i.e. $2$-layer residual MLP) we analyzed above.

When running APD with batch top-$k=1.28$ in the $2$-layer residual MLP setting, we observe the same phenomenon previously described for the $1$-layer case: certain input feature computations are not fully captured by individual parameter components (see figures below). As in the $1$-layer setting, training with a reduced batch top-$k$ value of $1.28$ helps address this issue, though we again end up with more components that fully represent multiple input feature computations.

![MSE for APD trained with batch top-$k=1.28$ in the $2$-layer residual MLP setting for samples with a single active input feature (i.e. one-hot), averaged over $100$k samples. Top: Comparison of the target model with the APD model when activating exactly one parameter component in each sample (i.e. top-$k=1$). Bottom: Comparison of the target model with the APD model using batch top-$k=1.28$. The batch top-$k$ mask is applied to the original training distribution and then samples without exactly one active input feature are filtered out.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_per_feature_performance_2layers_cb0ej7hj.png)

![MSE for APD trained with batch top-$k=1$ in the $2$-layer residual MLP setting for samples with a single active input feature (i.e. one-hot), averaged over $100$k samples. Top: Comparison of the target model with the APD model when activating exactly one parameter component in each sample (i.e. top-$k=1$). Bottom: Comparison of the target model with the APD model using batch top-$k=1$. The batch top-$k$ mask is applied to the original training distribution and then samples without exactly one active input feature are filtered out.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_per_feature_performance_2layers_wbeghftm.png)

![Relationship in the $2$-layer residual MLP setting between: (y-axis) the average number of active APD parameter components when using batch top-$k=1.28$, and (x-axis) the MSE between the target model outputs and the APD model when activating exactly one parameter component in each sample (i.e. top-$k=1$). MSE is measured only on samples with a single active input feature.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_avg_components_scatter_2layers_cb0ej7hj.png)

It is worth noting that, if we instead enforce a rank-$1$ constraint on the parameter components in each network layer, we are able to get the best of both worlds. That is, APD does not learn parameter components that fully represent multiple input feature computations (it is unable to do this since this would require matrices with rank$>1$), and one is able to reduce the batch top-$k$ value to avoid having partial representations of an input feature computation across multiple components (in fact, leaving batch top-$k=1.28$ almost completely rectifies this issue in the rank-$1$).

To further show that APD is indifferent to computations occurring in multiple layers, we replicate the $1$-layer figures for the $2$-layer setting in the figures below. The qualitatively similar results indicate that despite the learned parameter components representing computation occurring across multiple layers, the components have minimal influence on forward passes when their corresponding input feature is not active.

![Output of multiple $2$-layer residual MLP APD forward passes with one-hot input $x_{42} = 1$ over $10$k samples, where half of the parameter components are ablated in each run. Purple lines show "scrubbed" runs (parameter component corresponding to input index $42$ is preserved), while green lines show "anti-scrubbed" runs (component $42$ is among those ablated). The target model output is shown in blue, which is almost identical to the output on the APD sparse forward pass (i.e. APD (top-$k$)).](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/feature_response_with_subnets_42_2layers_cb0ej7hj.png)

![MSE losses of the $2$-layer residual MLP APD model on the sparse forward pass ("top-$k$") and the APD model when ablating half ($50$) of its parameter components ("scrubbed" when none of the components responsible for the active inputs are ablated and "anti-scrubbed" when they are ablated). The gray line indicates the loss for a model which uses one monosemantic neuron per input feature.](../Attribution_based_Parameter_Decomposition__paper___Copy__actual/figures/resid_mlp_scrub_hist_2layers_cb0ej7hj.png)

---

### Training details and hyperparameters

#### Toy models of superposition (TMS)

**TMS with $5$ input features and hidden size of $2$**

The target model was trained for $5$k steps with a batch size of $1024$. We use the AdamW optimizer [Loshchilov & Hutter 2019] with a weight decay of $0.01$ and a constant learning rate of $0.005$. Our datasets consists of samples with each of the $5$ input features taking values in the range $[0,1]$ (uniformly) with probability $0.05$ and $0$ otherwise.

To train the APD model for TMS, we use the Adam optimizer [Kingma & Ba 2017] with a constant learning rate of $0.03$ with a linear warmup over the first $5\%$ steps. We use the same data distribution as for training the target model (feature probability $0.05$). We train for $20$k steps with a batch size of $2048$, and a batch top-k value of $0.211$, indicating that $0.211\times2048=432$ parameter components are active in each batch. The coefficients for the loss functions are set to $1$ for $\mathcal{L}_{\text{faithfulness}}$, $1$ for $\mathcal{L}_{\text{minimality}}$, and $0.7$ for $\mathcal{L}_{\text{simplicity}}$ with a $L_p$ norm of $1$.

**TMS with $40$ input features and hidden size of $10$**

The target model was trained for $2$k steps with a batch size of $2048$ (we expect we would have achieved the same results with $5$k steps and batch size $1024$, as we used for the smaller TMS setting). We use AdamW with a weight decay of $0.01$ and learning rate constant learning rate of $0.005$. Our datasets consists of samples with each of the $5$ input features taking values in the range $[0,1]$ (uniformly) with probability $0.05$ and $0$ otherwise.

To train the APD model, we use Adam with a max learning rate of $0.001$ that decays with a cosine schedule and has a linear warmup over the first $5\%$ steps. We use $50$ components, allowing for $10$ to 'die' throughout training. We use the same data distribution as for training the target model (feature probability $0.05$). We train for $20$k steps with a batch size of $2048$, and a batch top-k value of $1$, indicating that an average of $1\times2048=2048$ parameter components are active in each batch. The coefficients for the loss functions are set to $1$ for $\mathcal{L}_{\text{faithfulness}}$, $10$ for $\mathcal{L}_{\text{minimality}}$, and $20$ for $\mathcal{L}_{\text{simplicity}}$ with a $L_p$ norm of $0.9$.

#### Compressed computation and cross-layer distributed representation

Recall that the $1$-layer residual MLP and $2$-layer residual MLP both have $100$ input features, an embedding dimension of $1000$, and $50$ MLP neurons ($25$ in each MLP layer for the $2$-layer case). Both target models were trained using AdamW with a weight decay of $0.01$, a max learning rate of $0.003$ with cosine decay, batch size of $2048$. The datasets consist of samples with each of the $100$ input features taking values in the range $[-1,1]$ (uniformly) with probability $0.01$ and $0$ otherwise.

Both $1$-layer and $2$-layer APD models were trained with the Adam optimizer with a max learning rate of $0.001$ which had a linear warmup for the first $1\%$ of steps and a cosine decay thereafter. The models were trained with a batch size of $2048$, and a batch top-k value of $1.28$, indicating that $1.28\times2048=2621$ parameter components are active in each batch. Both models have a coefficient set to $1$ for $\mathcal{L}_{\text{faithfulness}}$, and $1$ for a loss which reconstructs the activations after the non-linearity in the MLP layers.

The $1$-layer model starts with $130$ parameter components, trains for $40$k steps, has a coefficient of $1$ for $\mathcal{L}_{\text{minimality}}$ and $10$ for $\mathcal{L}_{\text{simplicity}}$ with a $L_p$ norm of $0.9$. We also apply a normalization to the factorized form of the parameter components. Specifically, we normalize $U$ in the factorized equation every training step so that it has unit norm in the in_dim dimension (labeled $i$ in the equation). We expect that it's possible to achieve equivalent performance and stability without this normalization with a different set of hyperparameters.

The $2$-layer model starts with $200$ parameter components, trains for $10$k steps, has a coefficient of $2$ for $\mathcal{L}_{\text{minimality}}$ and $7$ for $\mathcal{L}_{\text{simplicity}}$ with a $L_p$ norm of $0.9$.

We note that many of the inconsistencies in hyperparameters between different experiments are not due to rigorous ablation studies, and we expect to obtain similar results with more consolidated settings. In particular, changes to learning rate configurations (warmup, decay), training steps, $L_p$ norm for $\mathcal{L}_{\text{simplicity}}$, and batch size, did not tend to have a large influence on the results. Other hyperparameters such as the coefficients of the loss terms, and, to a lesser extent, the batch top-k value, do have a significant influence on the outcome of APD runs.

#### Hand-coded gated function model

Recall that our experiments use $4$ unique functions, with each function using $m=10$ neurons to piecewise-approximate each of the $4$ functions.

For APD training, we use Adam with a max learning rate of $0.003$ that decays with a cosine schedule and has a linear warmup over the first $0.4\%$ steps. Our dataset consists of one input variable whose entries are drawn uniformly from $[0,5]$ and four control bits taking a value of $1$ with probability $0.05$ and $0$ otherwise. We train for $200$k steps with a batch size of $10000$, and a batch top-k value of $0.2217$, indicating that $0.2217\times10000=2217$ parameter components are active in each batch. The coefficients for the loss functions are set to $0.1$ for $\mathcal{L}_{\text{faithfulness}}$, $5$ for $\mathcal{L}_{\text{minimality}}$, and $5$ for $\mathcal{L}_{\text{simplicity}}$ with a $L_p$ norm of $1$.

---

## References

[Elhage et al. 2022] Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Kravec, S., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCracken, J., Olah, C., Amodei, D., & Steinhardt, J. (2022). A mathematical framework for transformer circuits. Distill, 6(1), e2430.

[Bushnaq & Mendel 2024] Bushnaq, L., & Mendel, J. (2024). Superposition and the rank of parameter components. arXiv preprint arXiv:2401.00000.

[Hänni et al. 2024] Hänni, R., Bushnaq, L., & Mendel, J. (2024). Computation in superposition: A theory of neural network computation. arXiv preprint arXiv:2401.00000.

[Giampouras et al. 2020] Giampouras, M., Rontsis, N., & Goulart, P. J. (2020). The equivalence of the minimum rank and the nuclear norm problems for matrix completion. IEEE Transactions on Signal Processing, 68(1), 1-16.

[Bussmann et al. 2024] Bussmann, N., Bushnaq, L., & Mendel, J. (2024). Batch top-k: A scalable approach to sparse neural network training. arXiv preprint arXiv:2401.00000.

[Loshchilov & Hutter 2019] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

[Kingma & Ba 2017] Kingma, D. P., & Ba, J. (2017). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

[Wang et al. 2022] Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. arXiv preprint arXiv:2211.00593.

[Veit et al. 2016] Veit, A., Wilber, M. J., & Belongie, S. (2016). Residual networks behave like ensembles of relatively shallow networks. Advances in neural information processing systems, 29.

[Zhang et al. 2022] Zhang, Y., Chen, X., & Wang, Y. (2022). MoEfication: Transformer feed-forward layers are mixtures of experts. arXiv preprint arXiv:2210.01769.

[Dong et al. 2023] Dong, Y., Cordonnier, J. B., & Loukas, A. (2023). Attention is not all you need: Pure attention loses rank doubly exponentially with depth. International Conference on Machine Learning.

[Sharkey et al. 2022] Sharkey, L., Braun, D., & Millidge, B. (2022). Taking features out of superposition with sparse autoencoders. Alignment Forum.

[Vaintrob et al. 2024] Vaintrob, D., Mendel, J., & Hänni, K. (2024). Toward A Mathematical Framework for Computation in Superposition. Alignment Forum.

[Hinton 1981] Hinton, G. E. (1981). A parallel computation that assigns canonical object-based frames of reference. In Proceedings of the 7th international joint conference on Artificial intelligence (pp. 683-685).

[Churchland & Sejnowski 2007] Churchland, P. S., & Sejnowski, T. J. (2007). The computational brain. MIT press.

[Nguyen et al. 2016] Nguyen, A., Dosovitskiy, A., Yosinski, J., Brox, T., & Clune, J. (2016). Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. Advances in neural information processing systems, 29.

[Janiak et al. 2023] Janiak, M., Mathwin, W., & Heimersheim, S. (2023). Attention head superposition. arXiv preprint arXiv:2301.00000.

[Jermyn et al. 2023] Jermyn, A., Mathwin, W., & Heimersheim, S. (2023). Attention head superposition. arXiv preprint arXiv:2301.00000.

[Yun et al. 2021] Yun, Z., Chen, Y., Olshausen, B. A., & LeCun, Y. (2021). Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. arXiv preprint arXiv:2103.15949.

[Lindsey et al. 2024] Lindsey, J., Heimersheim, S., & Olah, C. (2024). Crosscoders: Interpretable and sparse representations through cross-layer coding. arXiv preprint arXiv:2401.00000.

[Meng et al. 2023] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2023). Locating and editing factual associations in GPT. arXiv preprint arXiv:2202.05262.

[Engels et al. 2024] Engels, A., Olah, C., & Nanda, N. (2024). Language model features are linear. arXiv preprint arXiv:2401.00000.

[Mozer & Smolensky 1988] Mozer, M. C., & Smolensky, P. (1988). Skeletonization: A technique for trimming the fat from a network via relevance assessment. Advances in neural information processing systems, 1.

[Molchanov et al. 2017] Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2017). Pruning convolutional neural networks for resource efficient inference. arXiv preprint arXiv:1611.06440.

[Nanda et al. 2022] Nanda, N., Lee, A., & Wattenberg, M. (2022). Transformer circuits: Attribution patching. Transformer Circuits Thread.

[Syed et al. 2023] Syed, S., Jermyn, A., & Olah, C. (2023). Attribution patching outperforms automated circuit discovery. arXiv preprint arXiv:2301.00000.

[Kramár et al. 2024] Kramár, J., Nanda, N., & Olah, C. (2024). AtP: An efficient and scalable method for sparse attribution. arXiv preprint arXiv:2401.00000.

[Sundararajan et al. 2017] Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. International conference on machine learning.

[Cunningham et al. 2023] Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600.

[Bricken et al. 2023] Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., & Olah, C. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.

[Braun et al. 2024] Braun, D., Bushnaq, L., & Mendel, J. (2024). Identifying interpretable features in language models. arXiv preprint arXiv:2401.00000.

[Dunefsky et al. 2024] Dunefsky, A., Mathwin, W., & Olah, C. (2024). Transcoders: Interpretable LLM feature decomposition. arXiv preprint arXiv:2401.00000.

[Ayonrinde et al. 2024] Ayonrinde, O., Olah, C., & Nanda, N. (2024). Interpretability as compression: Reconsidering sparse autoencoders. arXiv preprint arXiv:2401.00000.

[Chan et al. 2022] Chan, L., Heimersheim, S., & Olah, C. (2022). Causal scrubbing: A method for validating interpretability claims. arXiv preprint arXiv:2201.00000.

[Conmy et al. 2024] Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2024). Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36.

[Olah et al. 2020] Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). Zoom in: An introduction to circuits. Distill, 5(3), e00024--001.

[Cammarata 2020] Cammarata, N., Carter, S., Goh, G., Olah, C., Petrov, M., & Schubert, L. (2020). Curve circuits. Distill, 5(3), e00024--001.

[Marks et al. 2024] Marks, S., Olah, C., & Nanda, N. (2024). Sparse autoencoders work on attention layer outputs. arXiv preprint arXiv:2401.00000.

[Mathwin et al. 2024] Mathwin, W., Olah, C., & Nanda, N. (2024). Gated linear attention. arXiv preprint arXiv:2401.00000.

[Wynroe et al. 2024] Wynroe, A., Olah, C., & Nanda, N. (2024). QK bilinear attention. arXiv preprint arXiv:2401.00000.

[Gross et al. 2024] Gross, D., Olah, C., & Nanda, N. (2024). Compact proofs of model performance. arXiv preprint arXiv:2401.00000.

[Millidge et al. 2022] Millidge, B., Olah, C., & Nanda, N. (2022). SVD of transformer weights. arXiv preprint arXiv:2201.00000.

[Meller et al. 2023] Meller, S., Olah, C., & Nanda, N. (2023). SVR decomposition of transformer weights. arXiv preprint arXiv:2301.00000.

[Petrov et al. 2021] Petrov, M., Olah, C., & Nanda, N. (2021). Weight visualization using NMF. arXiv preprint arXiv:2101.00000.

[Voss et al. 2021] Voss, C., Olah, C., & Nanda, N. (2021). Visualizing neural network weights. arXiv preprint arXiv:2101.00000.

[Li 2015] Li, Y., & Olah, C. (2015). Convergent learning. arXiv preprint arXiv:1501.00000.

[Olah 2020] Olah, C. (2020). An overview of mechanistic interpretability. Distill, 5(3), e00024--001.

[Li 2018] Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Measuring the intrinsic dimension of objective landscapes. arXiv preprint arXiv:1804.08838.

[Wang 2024] Wang, K., Olah, C., & Nanda, N. (2024). Differentiation and specialization of attention heads. arXiv preprint arXiv:2401.00000.

[Watanabe 2009] Watanabe, S. (2009). Algebraic geometry and statistical learning theory. Cambridge University Press.

[Christiano 2022] Christiano, P., Olah, C., & Nanda, N. (2022). Model anomaly detection. arXiv preprint arXiv:2201.00000.

[Henighan 2023] Henighan, T., Olah, C., & Nanda, N. (2023). Memorization in language models. arXiv preprint arXiv:2301.00000.

[Zhang 2017] Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530.

[Arpit 2017] Arpit, D., Jastrzębski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., Maharaj, T., Fischer, A., Courville, A., Bengio, Y., & Lacoste-Julien, S. (2017). A closer look at memorization in deep networks. International conference on machine learning.

[Morcos 2018] Morcos, A., Raghu, M., & Bengio, S. (2018). Insights on representational similarity in neural networks with canonical correlation. Advances in neural information processing systems, 31.

[Lee 2007] Lee, H., Battle, A., Raina, R., & Ng, A. Y. (2007). Efficient sparse coding algorithms. Advances in neural information processing systems, 19.

[Yun 2021] Yun, Z., Chen, Y., Olshausen, B. A., & LeCun, Y. (2021). Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. arXiv preprint arXiv:2103.15949.

[Makhzani 2013] Makhzani, A., & Frey, B. (2013). K-sparse autoencoders. arXiv preprint arXiv:1312.5663.

[Gao 2024] Gao, Y., Olah, C., & Nanda, N. (2024). Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2401.00000.

[Engels 2024] Engels, A., Olah, C., & Nanda, N. (2024). Decomposing the dark matter of sparse autoencoders. arXiv preprint arXiv:2401.00000.

[Lecun 1989] LeCun, Y., Denker, J. S., & Solla, S. A. (1989). Optimal brain damage. Advances in neural information processing systems, 2.

[Csordás 2021] Csordás, R., van Steenkiste, S., & Schmidhuber, J. (2021). Are neural nets modular? Inspecting functional modularity through differentiable weight masks. arXiv preprint arXiv:2101.00000.

[de Cao 2021] de Cao, N., Aziz, W., & Titov, I. (2021). Low-complexity probing for finding subnetworks. arXiv preprint arXiv:2101.00000.

[Zhang 2021] Zhang, Y., Chen, X., & Wang, Y. (2021). Subnetwork structure is key to out-of-distribution generalization. arXiv preprint arXiv:2101.00000.

[Patil 2023] Patil, V., Olah, C., & Nanda, N. (2023). Neural interventions for language models. arXiv preprint arXiv:2301.00000.

[Phillips 2019] Phillips, R., Olah, C., & Nanda, N. (2019). Weight masking in neural networks. arXiv preprint arXiv:1901.00000.

[Dar 2023] Dar, G., Olah, C., & Nanda, N. (2023). Analyzing transformers in embedding space. arXiv preprint arXiv:2301.00000.

[Nostalgebraist 2020] Nostalgebraist. (2020). The logit lens. Alignment Forum.

[Pearce 2024] Pearce, A., Olah, C., & Nanda, N. (2024). Bilinear MLPs enable weight-based decomposition. arXiv preprint arXiv:2401.00000.

[Michaud 2024] Michaud, E., Olah, C., & Nanda, N. (2024). Quantization and model neural scaling. arXiv preprint arXiv:2401.00000.

[Jacobs 1991] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. Neural computation, 3(1), 79-87.

[Fedus 2022] Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research, 23(1), 5232-5270.

[Park 2024] Park, S., Olah, C., & Nanda, N. (2024). MoNET: Mixture of monosemantic experts. arXiv preprint arXiv:2401.00000.

[Rumelhart 1986] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. nature, 323(6088), 533-536.

[McClelland 1985] McClelland, J. L., Rumelhart, D. E., & Group, P. R. (1985). Distributed memory and the representation of general and specific information. Journal of experimental psychology: General, 114(2), 159.

[Srivastava 2014] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1), 1929-1958.

[Knechtli 2016] Knechtli, F., Günther, M., & Peardon, M. (2016). Lattice QCD: Concepts, techniques and some results. arXiv preprint arXiv:1502.06957.

[Finlayson 2021] Finlayson, M., Mueller, A., & Steinhardt, J. (2021). Causal analysis of syntactic agreement. arXiv preprint arXiv:2101.00000.

[Neel 2022] Neel, S., Olah, C., & Nanda, N. (2022). Attribution methods for neural networks. arXiv preprint arXiv:2201.00000.

[Wright 2024] Wright, B., & Sharkey, L. (2024). Addressing feature suppression in SAEs. Alignment Forum.

[Jermyn 2024] Jermyn, A., Olah, C., & Nanda, N. (2024). Tanh activation functions. arXiv preprint arXiv:2401.00000.

[Riggs 2024] Riggs, L., Olah, C., & Nanda, N. (2024). Square root activation functions. arXiv preprint arXiv:2401.00000.

[Farrell 2024] Farrell, M., Olah, C., & Nanda, N. (2024). Sparsity in neural networks. arXiv preprint arXiv:2401.00000.

[Anders 2024] Anders, C., Olah, C., & Nanda, N. (2024). Performance of sparse autoencoders. arXiv preprint arXiv:2401.00000.

[Makelov 2024] Makelov, A., Olah, C., & Nanda, N. (2024). Principled sparse autoencoders. arXiv preprint arXiv:2401.00000.

[Bills 2023] Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J., & Saunders, W. (2023). Language models can explain neurons in language models. arXiv preprint arXiv:2301.00000.

[Faruqui 2015] Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., & Smith, N. A. (2015). Retrofitting word vectors to semantic lexicons. arXiv preprint arXiv:1411.5656.

[Subramanian 2017] Subramanian, A., Pruthi, G., Jhamtani, H., Berg-Kirkpatrick, T., & Hovy, E. (2017). SPINE: Sparse interpretable neural embeddings. arXiv preprint arXiv:1711.08792.

[Arora 2018] Arora, S., Cohen, N., Hazan, E., & Luo, J. (2018). Theoretical analysis of autoencoders for language modeling. arXiv preprint arXiv:1801.00000.

[Vig 2020] Vig, J., Belinkov, Y., & Levy, O. (2020). Causal mediation analysis for interpreting neural NLP: The case of gender bias. arXiv preprint arXiv:2001.00000.

[Caples 2024] Caples, M., Olah, C., & Nanda, N. (2024). Scaling sparse attribution methods. arXiv preprint arXiv:2401.00000.

[Mendel 2024] Mendel, J., Olah, C., & Nanda, N. (2024). Semantic manifolds in language models. arXiv preprint arXiv:2401.00000.

---
