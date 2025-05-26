<h2 align="center">Deep Active Inference Agents for Delayed and Long-Horizon Environments</h2>  
<p align="center">
    <a href="https://yavaryeganeh.github.io/">Yavar Taheri Yeganeh</a>
    ·
    <a href="https://ise.rutgers.edu/mohsen-jafari">Mohsen Jafari</a>
    ·
    <a href="https://www.mecc.polimi.it/en/research/faculty/prof-andrea-matta">Andrea Matta</a>

---
![Deep_AIF_Agents](img/daif_arch.png)

With the recent success of world-model agents—which extend the core idea of model-based reinforcement learning by learning a differentiable model for sample-efficient control across diverse tasks—active inference (AIF) offers a complementary, neuroscience-grounded paradigm that unifies perception, learning, and action within a single probabilistic framework powered by a generative model. Despite this promise, practical AIF agents still rely on accurate immediate predictions and exhaustive planning, a limitation that is exacerbated in delayed environments requiring plans over long horizons—tens to hundreds of steps. Moreover, most existing agents are evaluated on robotic or vision benchmarks which, while natural for biological agents, fall short of real-world industrial complexity. We address these limitations with a generative–policy architecture featuring (i) a multi-step latent transition that lets the generative model predict an entire horizon in a single look-ahead, (ii) an integrated policy network that enables the transition and receives gradients of the expected free energy, (iii) an alternating optimization scheme that updates model and policy from a replay buffer, and (iv) a single gradient step that plans over long horizons, eliminating exhaustive planning from the control loop. We evaluate our agent in an environment that mimics a realistic industrial scenario with delayed and long-horizon settings. The empirical results confirm the effectiveness of the proposed approach, demonstrating the coupled world-model with the AIF formalism yields an end-to-end probabilistic controller capable of effective decision making in delayed, long-horizon settings without handcrafted rewards or expensive planning.

## Requirements

You need a Python environment with the following libraries (and other supporting ones):

```
torch>=2.1.1
numpy>=1.24.3
simpy>=4.0.1
```
