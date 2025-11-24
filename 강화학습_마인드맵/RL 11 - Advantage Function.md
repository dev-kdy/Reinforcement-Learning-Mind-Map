상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 10 - Value Function (V, Q)]], [[RL 09 - Policy (정책)]], [[RL 21 - Actor-Critic (구조)]], [[RL 22 - Advantage 기반 A2C·A3C]], [[RL 23 - Proximal Policy Optimization (PPO)]], [[RL 20 - Policy Gradient (기본 PG)]]

---

## What (정의)

Advantage Function은  
어떤 상태에서 특정 행동을 했을 때가  
그 상태에서의 평균적인 기대 가치보다  
얼마나 더 좋은지(또는 나쁜지)를 나타내는 함수다.

가장 전형적인 정의는 다음과 같다.

- `A(s, a) = Q(s, a) - V(s)`
    

여기서

- `V(s)`는 그 상태에서의 평균적인 기대 가치
    
- `Q(s, a)`는 그 상태에서 그 행동을 했을 때의 기대 가치
    

이기 때문에,  
`A(s, a)`는 “그냥 평균대로 했을 때보다 이 행동이 얼마나 더 나았는지”를 수치로 표현한다.

- `A(s, a) > 0`  
    상태 s에서 행동 a를 하면  
    평균보다 좋은 선택이라는 뜻
    
- `A(s, a) < 0`  
    평균보다 나쁜 선택이라는 뜻
    
- `A(s, a) ≈ 0`  
    그 상태에서 평범한 수준의 행동이라는 뜻
    

요약하면, Advantage는  
Value를 기준선으로 빼서  
“상대적인 이득”만 남긴 값이다.

---

## Why (배경/목적)

정책을 업데이트할 때  
그냥 Return이나 Q만 사용하면  
학습 신호의 분산이 커서  
업데이트가 들쭉날쭉해지기 쉽다.

Advantage를 쓰는 이유는 다음과 같다.

1. 기준선을 빼서 분산 감소
    
    - 같은 상태 s에서  
        모든 행동에 대해 공통적으로 나타나는 “기본적으로 이 상태가 좋은 정도”는  
        굳이 정책 업데이트에 영향을 줄 필요가 없다.
        
    - `Q(s, a)` 대신 `Q(s, a) - V(s)`를 쓰면  
        상태별 공통값 V(s)가 빠져서  
        순수하게 행동 간 차이만 남는다.
        
    - 이렇게 기준선을 빼 주면  
        Policy Gradient 신호의 분산이 줄어들어  
        학습이 더 안정적이 된다.
        
2. “평균보다 나았는지/못했는지”에 집중
    
    - Return이 크더라도  
        그 상태에서 당연히 나와야 하는 수준이라면  
        정책을 크게 바꿀 필요가 없다.
        
    - Advantage는  
        평균보다 특별히 좋았던 행동만 강하게 강화하고,  
        평균보다 나빴던 행동은 약하게 또는 반대로 조정하게 만들어 준다.
        
3. Actor-Critic 구조와 자연스럽게 맞물림
    
    - Critic이 V(s) 또는 Q(s, a)를 추정하고
        
    - Actor는 Advantage를 받아  
        그에 비례해 정책을 업데이트한다.
        
    - 이 구조에서 Advantage는  
        두 모듈을 연결하는 핵심 신호가 된다.
        

그래서 A2C, A3C, PPO 등  
현대 정책 기반 알고리즘 대부분은  
Return이나 Q 대신 Advantage를  
정책 업데이트의 가중치로 사용한다.

---

## How (활용)

### 1. 전형적인 정의와 근사

가장 기본 정의는 다음 두 가지 패턴으로 많이 쓴다.

1. `A(s, a) = Q(s, a) - V(s)`
    
    - Q와 V를 둘 다 근사할 수 있을 때  
        직접 이 차이를 계산한다.
        
2. `A(s, a) ≈ G_t - V(s_t)`
    
    - 에피소드 또는 trajectory에서 얻은 실제 Return `G_t`를 쓰고,
        
    - 그 시점의 상태 가치 추정치 `V(s_t)`를 빼서  
        Advantage를 근사한다.
        
    - A2C, A3C 등에서 자주 쓰는 형태다.
        

또는 TD 방식으로 한 스텝 Advantage를 정의하기도 한다.

- `A_t ≈ r_{t+1} + γ V(s_{t+1}) - V(s_t)`
    

이 값은 TD 오차와 같은 형태이며,  
한 스텝 동안 “예상보다 더 잘했는지/못했는지”를 나타내는 신호가 된다.

조금 더 고급으로는  
GAE(Generalized Advantage Estimation)처럼  
여러 스텝의 정보를 섞어  
부드럽고 분산이 낮은 Advantage 추정치를 만들기도 한다  
(보통 PPO와 함께 많이 등장).

---

### 2. Policy Gradient에서의 역할

기본적인 Policy Gradient에서는  
정책 θ를 업데이트할 때  
대략 다음과 같은 형태를 쓴다고 볼 수 있다.

- “좋은 행동을 했으면 그 행동의 확률을 늘리고,  
    나쁜 행동을 했으면 그 행동의 확률을 줄인다.”
    

이때 “얼마나 좋았는지/나빴는지”를 나타내는 가중치가  
바로 Advantage다.

정성적으로 보면

- Advantage가 큰 행동
    
    - 그 행동을 했을 때  
        예상보다 훨씬 좋은 결과를 냈으므로  
        그 행동의 확률이 크게 증가한다.
        
- Advantage가 작은 또는 음수인 행동
    
    - 평균 수준이거나 평균보다 나쁜 행동이므로  
        그 행동의 확률이 줄어들거나  
        거의 업데이트되지 않는다.
        

[[RL 21 - Actor-Critic (구조)]], [[RL 22 - Advantage 기반 A2C·A3C]], [[RL 23 - Proximal Policy Optimization (PPO)]] 모두  
정책 업데이트에서 Advantage를 곱해 쓰는 구조를 채택한다.

---

### 3. Actor-Critic 계열에서의 사용 패턴

Actor-Critic 구조에서

- Critic
    
    - V(s) 또는 Q(s, a)를 학습
        
- Actor
    
    - Policy π(a|s)를 학습
        

Advantage는 이 둘을 연결하는 브리지 역할을 한다.

대표 패턴을 정리하면 다음과 같다.

1. A2C / A3C
    
    - Critic이 V(s)를 추정
        
    - Advantage를 `A_t = G_t - V(s_t)` 또는  
        `A_t = r_{t+1} + γ V(s_{t+1}) - V(s_t)` 등으로 계산
        
    - Actor는 이 A_t를 곱해 Policy를 업데이트  
        (A_t가 클수록 해당 행동을 더 강하게 강화)
        
2. PPO
    
    - 중요도 비율 `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)`와  
        Advantage A_t를 함께 사용
        
    - `min( r_t A_t, clip(r_t, 1-ε, 1+ε) A_t )`  
        같은 형태의 목적을 최대화해  
        너무 큰 업데이트를 막으면서도  
        Advantage 방향으로 정책을 개선한다.
        

이처럼 Advantage는  
정책이 어느 방향으로, 어느 정도나  
업데이트되어야 하는지를 정해 주는  
핵심 신호로 사용된다.

---

### 4. 설계 및 실무 관점 요약

- Advantage의 핵심 아이디어
    
    - 절대적인 Return이나 Q 값보다  
        “이 상태에서 평균 대비 얼마나 더 좋았는지”만 보자는 것
        
- 기대 효과
    
    - 학습 신호의 분산을 줄여  
        Policy Gradient 업데이트를 더 안정적으로 만든다.
        
    - 좋은 행동, 나쁜 행동을 더 분명하게 구분해 준다.
        
- 구현 팁
    
    - Critic이 추정하는 V(s), Q(s, a)의 품질이 좋을수록  
        Advantage가 더 의미 있고 안정적으로 나온다.
        
    - Advantage를 계산할 때  
        너무 noisy하면 학습이 불안정해지므로  
        GAE 같은 기법으로 smoothing을 해 주는 것도 많이 사용된다.
        
- 정리
    
    - Value Function(RL 10)이 “절대적인 좋고 나쁨”을 알려 준다면,
        
    - Advantage Function(RL 11)은  
        “그 상태 기준으로 상대적으로 얼마나 더 좋았는지”를 알려 주는 역할을 한다.
        
    - 현대 Actor-Critic, PPO, A2C·A3C 계열을 이해할 때  
        사실상 필수 개념이라고 볼 수 있다.