# Interpreting Language Model Parameters

*TODO authors (affiliations, institutions)*

## Abstract

TODO abstract

## Introduction

Structure in the parameters of language models is responsible for their remarkable intelligence. The trainable parameters of these neural networks, in interaction with the architecture and dataset, learn to implement algorithms that we do not know how to design directly.
On the one hand, deep learning affords us the ability to build machines to solve tasks that otherwise resist engineering solutions, and incidentally creates objects that are of great scientific interest in their own right. On the other hand, it means that an increasing portion of our economy and daily lives depend on systems that we do not deeply understand (Bengio2026InternationalAISafety).

Our understanding of these systems is poor in part because it is unclear how best to decompose them into more fundamental units that we can study in relative isolation (mueller2024questrightmediatorhistory, sharkey2025openproblemsmechanisticinterpretability).
Naive choices of these units (such as neurons, attention heads, or whole layers) don't always map to individual, interpretable computations (hinton1981parallel, wei2015understandingintraclassknowledgeinside, nguyen2016multifacetedfeaturevisualizationuncovering, olah2017feature, janiak2023polysemantic, jermyn2023attention, yun2021sparse, lindsey2024crosscoders).
Consequently, methods such as sparse dictionary learning (yun2021sparse, Sharkey_2022, cunningham2023sparse, cunningham2023sparseautoencodershighlyinterpretable, bricken2023monosemanticity), transcoders (dunefsky2024transcodersinterpretablellmfeature, ameisen2025circuit), and mixtures of linear transforms (MOLTs) (oldfield2025towards, lindsey2025molts) were introduced to decompose datasets of neural activations, with the hope that they would identify units that approximate the network's underlying computational units. These methods, sometimes called *activation-based decomposition* methods, unfortunately suffer from a range of issues, including feature splitting (bricken2023monosemanticity, chanin2024absorptionstudyingfeaturesplitting) and unreliable level of mechanistic faithfulness (ameisen2025circuit) (See (sharkey2025openproblemsmechanisticinterpretability) for a more comprehensive review of these methods). Mechanistic unfaithfulness is suboptimal for activation-based methods' use in mechanistic analysis, and arises in part because these methods do not optimize for it. They approximate parts of the original network using functions of a different functional form as parts of the network. Their accounts of networks' computations are therefore not given in terms of the actual objects that are doing the computations---the network's parameters and its nonlinearities.

These issues motivate alternative approaches to mechanistic decomposition, including parameter decomposition methods (braun2025interpretabilityparameterspaceminimizing, bushnaq2025spd, chrisman2025identifyingsparselyactivecircuits), which give accounts of network function in terms of the components of the network's parameters that are used by the network on a given datapoint. *Ablation-based parameter decomposition methods* (braun2025interpretabilityparameterspaceminimizing, bushnaq2025spd) identify a set of parameter components where the set of components sum to the parameters of the target network, where the components are as simple as possible, and where as few components as possible are necessary to perform the same computations original network on any datapoint in a dataset. Parameter components are deemed 'necessary' if they cannot be ablated (including, crucially, partial ablations) on a given datapoint without adversely affecting output reconstruction error.

Parameter decomposition offers methods that, in toy models, can identify mechanisms that: are not aligned to e.g. neurons or individual attention heads; operate on representations in superposition; identify multidimensional mechanisms; and does not exhibit feature splitting. Moreover, unlike activation-based methods, where it has been challenging to use the same methods to decompose both attention layers and MLPs (kamath2025tracing, ameisen2025circuit, wynroe2024decomposing, ge2024localglobal), parameter decomposition methods are architecture-agnostic and can readily be applied to any architecture. In demonstration of this ability, beyond decomposing feedforward multi-layer perceptrons (MLPs), previous work has used ablation-based parameter decomposition to identify induction heads in a transformer trained on a toy model of induction (christensen2025decomposition).

Ablation-based parameter decomposition methods thus promise solutions to many of the issues of activation-based decomposition methods. However, several shortcomings remain with existing methods. While the most recent parameter decomposition method, Stochastic Parameter Decomposition (SPD) (bushnaq2025spd) is more scalable than its predecessor (Attribution-based Parameter Decomposition (braun2024identifying)), it has so far only been extensively studied in toy settings. While some work has applied SPD to a single layer of GPT2-small (christensen2025decomposition), no application of SPD so far has measured key metrics that would be necessary to ensure mechanistic faithfulness, such as output reconstruction under adversarial (rather than only stochastic) ablations.
Additionally, previous implementations of SPD have been partially incomplete: Attribution-based parameter decomposition (braun2025interpretabilityparameterspaceminimizing) decomposed networks into full vectors in parameter space, which span all parameters in the model. But SPD decomposes them into rank-one matrices, which are limited only to single parameter matrices. A full implementation of SPD requires a *post hoc* clustering step to combine multiple rank-one matrices into full vectors in parameter space, but previous work left this clustering step implicit (bushnaq2025spd). Finally, another issue is that previous work omitted analyses of the nonlinear interactions between parameter components, which would be crucial for assessing how useful parameter decomposition methods are for interpretability.

In this work, we introduce a parameter decomposition method that resolves all of these issues, called *ad**V**ersarial **P**arameter **D**ecomposition* (**VPD**), which builds heavily on the method introduced by (bushnaq2025spd) but has several important changes, which together make it more mechanistically faithful and scalable to larger models than decomposed in previous work (see Method section). Here, we decompose a small language model (XXM parameters) trained on the Pile (gao2020pile). We find parameter components that are highly interpretable (see Results section), both in terms of the dataset examples that they activate on and how they interact with other components to produce specific behaviors. We compare the parameter components that we find to the objects found by other decomposition methods, such as sparse autoencoder (SAE) latents and cross-layer transcoder (CLT) latents and find that they explain more of the target model's performance using an equivalent number of active components; exhibit less feature splitting; have comparable or greater interpretability; and are more mechanistically faithful. Furthermore, we analyze the nonlinear interactions between parameter components. We demonstrate that complex nonlinear interactions are much rarer than would be expected by chance, despite not being a property our method optimizes for directly, suggesting that it reflects an underlying computational simplicity in the target model itself. Finally, we demonstrate that our method identifies network components that [[[have better practical utility than alternative methods, such as 'are better for steering according to benchmarks', 'can be used to remove memorized datapoints', and 'even permit direct editing of the model parameters in interpretable ways' todo]]].


## Method - Adversarial Parameter Decomposition

Our method, VPD, builds heavily on SPD (bushnaq2025spd), but we do not assume familiarity with that work or its predecessor (braun2025interpretabilityparameterspaceminimizing). In this section, we introduce ablation-based parameter decomposition methods from scratch and highlight key differences between VPD and prior methods in this class.

We would like to decompose a neural network's parameters into the mechanisms that it uses to compute its behavior. To do this, we leverage the observation that networks appear not to use all of their parameters simultaneously on every datapoint (veit2016residual, zhang2022moefication, dong2023attention). This may happen, for instance, for parts of the parameters that are orthogonal to the activations on that datapoint, or if the activations fail to 'activate' a given ReLU neuron. If particular parameters are unused by the network on a particular datapoint, we should be able to ablate them (including partially) on that datapoint without adversely affecting the network's output. Ablation-based parameter decomposition methods thus aim to decompose networks into a set of vectors in parameter space called *parameter components* that are **faithful** (they sum to the parameters of the target model), **minimal** (as few as possible are required to replicate the network's outputs), and **simple** (components should each involve as little computation machinery as possible).

Suppose we have a neural network $f(x;\theta)$. We would like to decompose its parameter vector $\theta$ into a sum of parameter components $\theta = \sum_i \theta_i$ with the above properties. It would be computationally expensive to decompose models into whole parameter vectors, each of which would have a memory cost equivalent to the target model. Although its parameters $\theta$ can be expressed as a single large vector, they are more commonly conceptualized as a set of matrices $\theta = \{W_1, \dots , W_L\}$. As in (bushnaq2025spd), we decompose individual matrices into sums of rank-one matrices called *subcomponents*, each parametrized as an outer products of two vectors: $W_l \approx \sum_{c} \vec{U^l_c} \vec{V_c^{l  \top}} = U_l V_l^\top$, where there may be more subcomponents than rows and columns in the matrix. Although a single subcomponent explicitly parameterizes only a single weight matrix, it implicitly parametrises a full parameter vector if we assume it takes values of $0$ in every other weight matrix. It is therefore possible to combine these subcomponents into full parameter components by clustering them together. Previous work left this clustering implicit, whereas we introduce an explicit clustering method (see appendix on clustering method).

Optimizing for faithfulness is straightforward. As in (bushnaq2025spd) and (braun2025interpretabilityparameterspaceminimizing), we simply penalize the mean-squared error between the target model parameters and the sum of the subcomponents:

$$
\mathcal{L}_{\text{faithfulness}}=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}{\left( W^{l}_{i,j}- \sum^C_{c=1} U^l_{i,c} V^l_{c,j}\right)}^2
$$

Optimizing for minimality and simplicity are more involved. They require estimating whether each parameter component was 'required' to replicate the network's output on a given datapoint. And they require a notion of how well the 'required' subcomponents have reconstructed the network's output.

### Training a causal importance function requires an importance minimality loss and output reconstruction loss

Ablation-based parameter decomposition methods contend that a parameter component is 'required' if it cannot be ablated without affecting the model's output on that datapoint. As in (bushnaq2025spd), we train a *causal importance function* $\Gamma: X \rightarrow [0,1]^{C \times L}$ to predict how ablatable each subcomponent is on a given datapoint. As in (bushnaq2025spd), we implement $\Gamma$ as a neural network, though we use a different architecture (see appendix for details).

We want our causal importance function to output *causal importance values* $g^l_c(x)\in[0,1]$ for each subcomponent $(c)$ of weight matrix $l$ on a given datapoint $x$. If $g^l_c(x) = 0$, then that subcomponent should be fully or partially ablatable without affecting the output. If $g^l_c(x) = 1$, then it should not be possible to ablate that component without affecting the model's output on that datapoint. We want our causal importance values to predict the maximal extent of the ablatability of each subcomponent. Otherwise, the causal importance function could output a value of $1$ for every subcomponent on every input. We must therefore train the causal importance values $g^l_c(x)$ to take minimal values. We therefore use an importance minimality loss:

$$
\mathcal{L}_{\text{importance-minimality}}=\sum^L_{l=1}\sum^C_{c=1} \vert g^l_c(x) \vert^p
$$

where $p>0$.

We cannot simply use causal importance values as masks to ablate our subcomponents. They must satisfy a much stricter criterion: We should be able to use our causal importance values to create a continuous set of masks such that we can completely or partially mask our subcomponents while leaving the output unaffected. Formally, suppose for each subcomponent we have a scalar $r^l_c \in [0, 1]$. For all values of $r^l_c$ in that interval, we want to find subcomponents such that, when we create a (partial) ablation mask $m^l_c(x,r) :=g^l_c(x)+(1-g^l_c(x))r^l_c$ for that subcomponent, the outputs of the network are approximately the same:

$$
W'^l_{i,j}(x,r):=\sum^C_{c=1} U^l_{i,c} m^l_c(x,r) V^l_{c,j}
$$

$$
\forall r: f(x\vert W'^1(x,r),\dots,W'^L(x, r))\approx f(x\vert W^1,\dots,W^L)
$$

We can potentially use an output reconstruction loss to train the masked model $f(x\vert W'^1(x,r),\dots,W'^L(x, r))$ to approximate the target model's output $f(x\vert W^1,\dots,W^L)$. Unfortunately, we must do this for all values of $r\in {[0,1]}^{C\times L},$ which is a high dimensional continuous interval, making such a loss impossible to compute exactly. However, a key insight of (bushnaq2025spd) is that it is possible to *approximately* minimize reconstruction loss on all values in that interval.

The approximation can be achieved by taking a finite number $S$ of uniform random samples $r^{l,(s)}_{c} \sim \mathcal{U}(0,1)$, use those samples to create stochastic masks $m^l_c(x, g^l_c(x)) \sim \mathcal{U}(g^l_c(x), 1)$, and minimize reconstruction loss on that finite number of samples. This leads to the *stochastic reconstruction loss*:

$$
\mathcal{L}_{\text{stochastic-recon}}=\frac{1}{S}\sum^S_{s=1}D \left( f(x\vert W'(x,r^{(s)})),f(x\vert W) \right)
$$

where $D$ is an appropriate divergence measure in the space of model outputs, such as KL-divergence or mean squared error.

Together, the importance minimality loss and stochastic reconstruction should optimize parameter components to be able to replicate the target model's outputs while using as few parameter components as possible. We use these losses in our work. However, scaling ablation-based parameter decompositions to language models revealed several pathologies that were missed by (bushnaq2025spd). In the next sections, we introduce several other constraints to the optimization to address these issues.

### VPD optimizes for an even stricter criterion that SPD missed: Adversarial ablatability

The above stochastic loss does, in the limit of infinite samples, approximate the desired quantity in the subcomponents equation above. However, notice that it requires that the masked model approximates the target model for all possible values of $r$. This means that even if we adversarially---rather than merely randomly---sample $r$ in order to maximize reconstruction error, we shouldn't be able to find values where reconstruction error is high.

However, if we train using stochastic reconstruction loss (and the other losses), we find that adversarial sampling can find values of $r$ that have very high reconstruction loss (TODO: experiment figure), which is not permitted under the subcomponents equation. VPD therefore introduces an adversarial loss to help ensure this property:

$$
\mathcal{L}_{\text{adversarial-recon}}=\frac{1}{S} \sum^S_{s=1} \max_{r^{(s)}} D \left( f(x\vert W'(x,r^{(s)})),f(x\vert W) \right)
$$

### VPD better optimizes for simplicity than SPD

One of the reported benefits of SPD (bushnaq2025spd) over APD (braun2025interpretabilityparameterspaceminimizing) was that SPD used rank-one subcomponents, and where as few of these rank-one subcomponents are necessary to reconstruct the output. The authors believed this meant that SPD did not need dedicated losses to optimize for 'simplicity' (defined earlier as 'using as little computational machinery as possible'), whereas (braun2025interpretabilityparameterspaceminimizing) did. Their optimism was misplaced. Some rank-one solutions are 'simpler' than others. To see how this is possible, reflect on the possibility that it is possible to add multiple rank-one mechanisms together and for their sum also to be rank-one as long as either their right or left singular vectors are equal (TODO: potentially a diagram/figure illustrating the point). We observed indications that some SPD decompositions suffered from this failure mode: Sometimes subcomponents seemed to be involved in multiple (usually two) unrelated computations, which depended on whether the activations had strong positive or negative inner products with subcomponents' right singular vectors (see Appendix).

We therefore needed an additional loss to incentivize rank-one subcomponents to separate into subcomponents that are involved only in one type of computation to further encourage the 'simplicity' of parameter components (beyond the extent to which the importance minimality loss and rank-one constraint already encourage aspects of 'simplicity', as outlined in (bushnaq2025spd)). We settled on a loss function, called $\mathcal{L}_{\text{frequency-minimality}}$, which helps to ensure that parameter components are involved in only one kind of computation:

$$
\mathcal{L}_{\text{frequency-minimality-TODO}}=\sum^L_{l=1}\sum^C_{c=1} \vert g^l_c(x) \vert^p\log_2(1+\sum_{x'} \vert g^l_c(x') \vert^p)
$$

Although we are not confident that our choices of loss function are optimal for achieving minimality and simplicity, we remark that the new loss function has an interesting symmetry with the existing importance minimality loss: Where $\mathcal{L}_{\text{importance-minimality}}$ encourages datapoint in the training dataset to activate as few subcomponents as possible, the new loss encourages subcomponents to activate on as few datapoints in the training dataset as possible. This difference is subtle but important. It creates a new tradeoff during training: The importance minimality coefficient encourages the decomposition to have few components overall (while still being able to reconstruct the output), while the frequency minimality encourages it to have more components. We note that this tradeoff avoids creating the same problem as 'feature splitting' due to the tradeoffs between the reconstruction, importance minimality, and frequency minimality losses (TODO: results section reference).

*TODO: potentially figure illustrating the effects of the two losses on causal importances. TODO experiment: showing CIs on real data, where y axis is C and data index is on X axis, where the C activations are hierarchically clustered - should show that each component activates on fewer parameter components and there are fewer 'bias-like' components.*

These two losses represent the key differences in our approach compared with (bushnaq2025spd). However, there are several other, smaller differences that do not fundamentally change the method but that we found helpful for decomposing language models. For full details of our method, see Appendix.


## Results

### The language model that we decomposed

We trained a two-layer XXM parameter decoder-only transformer language model on The Pile (gao2020pile). It uses standard multihead attention layers (vaswani2017attention) and uses MLPs with a GELU activation function (hendrycks2016gelu), RMSNorm (zhang2019rootmeansquarelayer) applied to the inputs of the attention and MLP layers (xiong2020layernormalization) and before the final unembedding layer. It was trained for XTODO steps using the Adam optimizer (kingma2017adam). It achieves a final loss of XTODO which is similar to TODO-reference-model. The residual stream width and other hyperparameter choices can be found in the table below.

| Hyperparameter | Value |
|---|---|
| Residual stream $d_{\text{model}}$ | XXM |
| Layers | 2 |
| Number of attention heads | TODO |
| Attention head dimension | TODO |
| Learning rate | $Xe-3$ TODO |
| Adam $\beta_1$ | TODO |
| Adam $\beta_2$ | TODO |
| TODO | TODO |

*Table: Hyperparameters for our XXM Parameter Language Model*

Although most of the results in this paper focus on our decomposition of this language model, we show that VPD achieves canonical decompositions of the toy models decomposed in (braun2025interpretabilityparameterspaceminimizing) and (bushnaq2025spd) in the appendix.

It is worth noting that transformer models share parameters at each sequence index. Despite these shared parameters, different sequence indices usually have different activations, and therefore usually do different computations. Our causal importance functions therefore output different causal importance values for each sequence position, thus activating different sets of parameter components at different points in the sequence.

### Decomposing our language model: A recipe for VPD

### Parameter components approximate the target model well

*[Figure: Sample figure caption.]*

### The decomposition model behaves similarly to the target model

### Decomposition results are consistent for different seed

### Parameter components are highly interpretable

### Parameter components are mechanistically faithful

### Parameter components interact nonlinearly, but in an interpretable way

### Case studies: Interpretability in language model parameter space

#### Case study 1

#### Case study 2

#### Case study 3


## Discussion

### Related work

### Limitations

### Future work

### Conclusion

---

## Appendix

## Methods Details

TODO: CI function details, p-annealing, frequency aux penalty, adversarial mask computation

<!-- TODO: Full discussion of training details
- Adversarial reconstruction loss
- Importance function architecture
- P-annealing schedule
- Frequency penalty -->
