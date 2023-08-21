# diffusion-models

Diffusion models 논문 요약

1. **DDPM (Denoising Diffusion Probabilistic Models)**   [paper](https://arxiv.org/abs/2006.11239)  
   - assumption : gaussian noise, markov process, $\Sigma_{\theta} = \sigma_t^2 \mathbf{I}$  
   - objective function : variational bound 중 $L_{1:T-1}$ term, $L_simple$ 을 optimizing 하는 것과 유사
     $$L_{\text{simple}}(\theta) \coloneqq \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_{\theta} \left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|^2 \right]$$

   - goal : forward process가 given 일 때 reverse process $\mu^{\tilde}_{t}$, $\Sigma^{\tilde}_{t}$ fitting를 $\mu_{\theta}$, $\Sigma_{\theta}$ fitting  
            이 때 $\Sigma_{\theta}$ 는 fixed(not training)
   - 해결한 문제 : training 할 때는 $x_0$의 정보를 사용하나 sampling 할 때는 $x_0$의 정보를 사용하지 않는다. $x_t$를 통해 $x_0$를 추정해야 한다. gaussain noise가 iid 이기 때문에 $x_t$와 $\epsilon$ 을 안다면 $x_0$를 추정할 수 있다.
   - architecture : u-net


2. **Improved Denoising Diffusion Probabilistic Models**   [paper](https://arxiv.org/abs/2102.09672)
   - goal : improving log likelihood
   - improving log likelihood
      - $\Sigma_{\theta}$ term 을 training 하지 않았었는데 이를 구하기 위해서 $L_{hybrid}$ 이용 (기존 $L_simple$ 에는 \\
        $L_{hybrid} = L_{simple} + \lambda L_{vlb} \qquad (\lambda = 0.001)$\\
      - $Sigma_{\theta}(x_t, t) = \exp (\math

        
