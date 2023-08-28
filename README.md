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


4. **Generative Modeling by Estimating Gradients of the Data Distribution**   [paper](https://arxiv.org/abs/1907.05600)
   - goal : reverse SDE에서 유도된 $\nabla_x \log p(x)$ 를 효과적으로 score matching 시키는 방법
   - advantage : no adversarial training, no surrogate losses, no sampling from the score network during training
   - Score matching for score estimation : 효과적인 방식으로 $s_{\theta}(x) \sim \nabla_x \log p(x)$ 예측
      - trace based : $tr(\nabla_x s_{theta}(x))$ 이용; 하지만 high demensional data 일 때 $tr(\nable_x s_{theta}(x))$를 구하는 것은 힘들다
      - sliced score matching : $tr(\nable_x s_{theta}(x))$ 대신 $v^t \nabla_x s_{\theta}(x) v$ 이용 (v : multivariate standard normal vector); 직접 trace에 접근하지 않아도 되어 efficient

   - Sampling with Langevin dynamics : $\tilde{x}_t = \tilde{x}_{t-1} + \frac{\epsilon}{2} \nalba_x \log p(\tilde{x}_{t-1}) + \sqrt{\epsilon} z_t$; 이 때 $s_{\theta} (x)$를 $\nabla_x \log p(x)$ 대신 사용

   - Problem (Vanilla Sampling with Langevin dynamics)
      - under the manifold hypothesis, p_data가 존재하지 않는 x 에서 $s_theta(x)$가 정의되지 않음
      - low data density 영역에서의 부정확
      - mixture data distribution에 대한 분별능력 X
         - e.g. $p_data = 0.2 \mathcal{N} ((0,0), I)) + 0.8  \mathcal{N} ((1,1), I))$ 라고 하면, 이상적인 경우 20%는 $\mathcal{N} ((0,0), I))$, 그리고 80%는 $\mathcal{N} ((1,1), I))$로 분류하기를 원함; 하지만, 임의의 점에서 시작한다면 거의 50:50으로 분류 (why? $\mathcal{N} ((0,0), I))$ 근방에서는 $(0,0)$ 방향으로 gradient가 끌어당기는 힘이 더 강하고, $\mathcal{N} ((1,1), I))$ 근방에서는 $(1,1)$ 방향으로 gradient가 끌어당기는 힘이 더 강하기 때문에) <br />
           ![image](https://github.com/ddolmaeng/diffusion-paper-summary/assets/112860653/c5c1866d-8cf8-4332-b75d-67202758a27c)


   - Solution (perturbed data distribution)
      - ${\sigma_i}^{L}_{i=1}$, $\frac{\simga_1}{\sigma_2} = \cdots = \frac{\sigma_{L-1}}{\sigma_L} > 1$ 을 만족하게 sequence 잡은 후
      - $q_sigma(x) = \int p_{\text{data}}(t) \mathcal{N} (x | t, \sigma^2 I) dt$ 로 각 step 마다 점점 perturbed noise가 작아지게 data distribution을 setting 한다.
      - 대신 추정해야하는 $s_{\theta} (x)$ 도 더이상 $x$에만 영향을 받지 않고 $\sigma$에 영향을 받게 setting. $s_{\theta} (x, \sigma)$

   - denoising score matching : $s_theta(x,t) \sim \nabla \log q_{\sigma_t}(x)$
      - $\arg \min_{\theta} \Sigma_{t=1}^{T} \lambda(\sigma_t) \mathbb{E}_{q_{\sigma_t}} [\|s_{\theta}(x, t) - \nabla \log q_{\sigma_t} (x_t)\|_2^2]$
      - $\lambda(\sigma_t)$ : coefficient, 논문에서는 $\lambda(\sigma) = \sigma^2$ 이용 (why? variance가 $\sigma^2 I$인 정규분포로 perturbed 했기 때문에 $\|s_{\theta} (x, \sigma)\|_2 \propto 1/{\sigma}$)

   - algorithm <br />
     ![image](https://github.com/ddolmaeng/diffusion-paper-summary/assets/112860653/99cfb858-de57-464c-b165-861616a6170f)



7. **Score-Based Generative Modeling through Stochastic Differential Equations**   [paper](https://arxiv.org/abs/2011.13456)
   - goal : 













