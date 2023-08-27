# diffusion-models

Diffusion models 논문 요약

1. **DDPM (Denoising Diffusion Probabilistic Models)**   [paper](https://arxiv.org/abs/2006.11239)  
   - assumption : gaussian noise, markovian process, $\Sigma_{\theta} = \sigma_t^2 \mathbf{I}$  
   - objective function : variational bound 중 $L_{1:T-1}$ term -> $L_simple$ 을 optimizing 하는 것과 유사
   - $L_{\text{simple}}(\theta) \coloneqq \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_{\theta} \left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|^2 \right]$

   - goal : forward process가 given 일 때 reverse process $\mu^{\tilde}_{t}$, $\Sigma^{\tilde}_{t}$ fitting를 $\mu_{\theta}$, $\Sigma_{\theta}$ fitting; 이 때 $\Sigma_{\theta}$ 는 fixed(not training)
   - 해결한 문제 : training 할 때는 $x_0$의 정보를 사용하나 sampling 할 때는 $x_0$의 정보를 사용하지 않는다. $x_t$를 통해 $x_0$를 추정해야 한다. gaussain noise가 iid 이기 때문에 $x_t$와 $\epsilon$ 을 안다면 $x_0$를 추정할 수 있다.
   - architecture : u-net


2. **Improved Denoising Diffusion Probabilistic Models**   [paper](https://arxiv.org/abs/2102.09672)
   - goal : improving DDPM
   - improving log likelihood
      - learning $\Sigma_{\theta}$
         - $\Sigma_{\theta}$ term 을 training 하지 않았었는데 이를 구하기 위해서 $L_{hybrid}$ 이용 (기존 $L_{simple}$ 에는 $\Sigma_{\theta} 와 관련된 항이 존재 X (L_{simple} 에서 \sigma_t term 을 제외하였음); $L_{hybrid} = L_{simple} + \lambda L_{vlb} \qquad (\lambda = 0.001)$
         - $\Sigma_{\theta}(x_t, t) = \exp(v \log \beta_t + (1 - v) \log \tilde{\beta}_t)$
           $v$를 fitting, 즉 $Sigma_theta$도 자유도를 가진다.
   
      - improving the noise schedule
         - forward process 기준 마지막으로 가면 갈수록 너무 nosie의 비율이 높아 noise를 천천히 추가함
         - linear schedule -> cosine schedule
           
      - reducing gradient noise
         - $L_{vlb} = L_0 + \cdots + L_T$ 의 noise가 크다고 판단해 importance sampling 진행

   - improving sampling speed
      - $L_{simple}$ 보다 적은 step에서 FID score가 안정화 되는 것을 관측할 수 있다.
      - 하지만 DDIM도 사실 비슷한 성능

3. **Denoising Diffusion Implicit Models**   [paper](https://arxiv.org/abs/2010.02502)
   - goal : assumption(non-markovian) 변경을 통한 DDPM 가속화(적은 step으로도 학습 가능하게)
   - 가속화 하는 법 : 병렬 처리가 가능한 구조 ($x_{t-1}$이 $x_{t}$에 cascade하지 않게, 바꿔서 말하면 deterministic 하게)
   - Assumption
   - $q_{\sigma}(X_{1:T}|X_0) = q_{\sigma}(X_1|X_0) \Pi_{t=2}^{T} q_{\sigma} (X_{t}|X_{t-1}, X_0) = q_{\sigma} (X_T|X_0) \Pi_{t=2}^{T} q_{\sigma} (X_{t-1}|X_{t}, X_0)$
   - $q_{\sigma} (X_{t-1}|X_{t}, X_0)$ is gaussian (이 때 mu 는 DDPM과 똑같은 세팅, 그리고 여기서 variance를 0으로 만들어준다면 deterministic한 결과를 얻을 수 있음, DDPM과 똑같은 variance를 넣는 것도 가)

   - Question : 그러면 왜 deterministic한 result를 얻기 위해서는 왜 non-markovian process를 사용하여야하는가?
   - Answer : markovian process를 가정하면 $q(x_{t-1}|x_t, x_0)$ 의 분산을 0으로 컨트롤 할 수 없다. 이 때 분산을 0으로 만들어주기 위해서는 $\alpha_t$ 를 1로 만들어야하는데 이 의미는 forward process에서 어떠한 noise도 더하지 않겠다는 의미

   - sampling acceleration : sampling 과정에서 sub-sequecne를 이용하게 하면 적은 step으로도 sampling 가능

4. **Score-Based Generative Modeling through Stochastic Differential Equations**   [paper](https://arxiv.org/abs/2011.13456)
   - goal : 
