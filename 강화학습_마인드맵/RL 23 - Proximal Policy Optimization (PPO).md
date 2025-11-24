상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 20 - Policy Gradient (기본 PG)]], [[RL 21 - Actor-Critic (구조)]], [[RL 11 - Advantage Function]], [[RL 22 - Advantage 기반 A2C·A3C]]

---

## What (정의)

PPO(Proximal Policy Optimization)는  
정책이 한 번의 업데이트에서 너무 크게 변하지 않도록 제약을 거는,  
실용적인 Actor-Critic 기반 Policy Gradient 알고리즘이다.

조금 더 풀어서 말하면 다음과 같다.

- 기본 뼈대는 [[RL 21 - Actor-Critic (구조)]]와 동일하다.  
    Actor는 정책 πθ(a|s)를, Critic은 가치 함수 V(s)를 학습한다.
    
- 정책 업데이트 시, 이전 정책과 현재 정책의 비율 r_t를 이용해  
    정책 변화량을 제한하는 특별한 목적 함수(clipped objective)를 사용한다.
    
- [[RL 11 - Advantage Function]] A(s, a)를 가중치로 사용해  
    좋은 행동의 확률은 늘리고 나쁜 행동의 확률은 줄이되,  
    한 번에 과하게 바뀌지 않도록 조절한다.
    

요약하면, PPO는  
“정책 Gradient를 쓰되, 너무 멀리 점프하지 않도록  
안전 범위 안에서만 업데이트하는 Actor-Critic 알고리즘”이다.

---

## Why (배경/목적)

정책 Gradient 계열 알고리즘([[RL 20 - Policy Gradient (기본 PG)]], [[RL 22 - Advantage 기반 A2C·A3C]] 등)는  
기본 개념은 단순하지만 다음과 같은 문제가 있다.

1. 큰 업데이트 한 번이 정책을 망가뜨릴 수 있다
    
    - 학습률이 조금만 커져도  
        한 번의 gradient step으로 정책이 크게 변해  
        지금까지 배운 것을 무너뜨리거나,  
        성능이 급락하는 현상이 자주 발생한다.
        
    - 특히 고차원 정책, 큰 모델, 복잡한 환경에서는  
        이 문제가 더 심해진다.
        
2. TRPO는 안정적이지만 구현이 복잡하다
    
    - 이전 연구인 TRPO(Trust Region Policy Optimization)는  
        정책 변화량에 대한 제약(Trust Region)을  
        이론적으로 정교하게 다루지만,  
        2차 정보와 제약 최적화를 포함해 구현이 어렵고 무겁다.
        

PPO는 이 두 문제를 동시에 겨냥한다.

- TRPO의 “정책을 너무 멀리 움직이지 말자”라는 아이디어를 유지하면서
    
- 실제 구현은 간단한 클리핑(clipping)과 1차 Gradient만으로 해결하도록 설계했다.
    
- 그 결과  
    안정성과 성능은 높고,  
    구현 난이도는 낮은 “현대 표준” RL 알고리즘 역할을 하게 되었다.
    

그래서 복잡한 게임 환경, 로봇 제어, RLHF 같은 대규모 환경에서  
PPO가 사실상 기본 선택지처럼 널리 쓰인다.

---

## How (활용)

### 1. 기본 아이디어: 비율 r_t와 클리핑

PPO의 정책 업데이트 핵심은  
이전 정책과 새 정책의 비율 r_t를 사용하는 것이다.

- r_t = πθ(a_t | s_t) / πθ_old(a_t | s_t)  
    라고 생각하면 된다.
    
- r_t가 1이면  
    새 정책과 옛 정책이 그 행동에 대해 같은 확률을 주는 상태이고,
    
- r_t가 1보다 크면  
    그 행동의 확률이 커진 것,  
    1보다 작으면 줄어든 것이다.
    

정책 Gradient의 기본 형태는  
r_t와 Advantage A_t를 곱한 값을  
크게 만들도록 θ를 업데이트하는 것이다.

- A_t > 0인 행동은  
    r_t를 1보다 크게 만들어  
    그 행동을 더 자주 선택하게 만들고,
    
- A_t < 0인 행동은  
    r_t를 1보다 작게 만들어  
    그 행동의 선택 확률을 줄이게 만든다.
    

하지만 r_t가 너무 커지거나 너무 작아지면  
정책이 한 번에 크게 변하게 된다.  
그래서 PPO는 r_t를 특정 범위에서만 허용하도록  
“clipped objective”를 사용한다.

- 개념적으로는  
    r_t를 [1 − ε, 1 + ε] 범위로 잘라 내고  
    Advantage에 불리한 방향의 지나친 변화는  
    더 이상 이득을 주지 않도록 막는다.
    
- 이로써  
    한 번의 학습 스텝에서  
    정책이 과도하게 업데이트되는 것을 제한한다.
    

정리하면

- r_t는 “정책이 얼마나 바뀌었는지”를 측정하는 지표이고
    
- 클리핑은 “너무 많이 바뀌면 그 이상은 이득이 없다”는  
    안전 범위 역할을 한다.
    

---

### 2. Actor 업데이트 흐름 (개념 단계)

PPO 기반 Actor-Critic 학습을 개념적으로 정리하면 다음과 같다.

1. 데이터 수집
    
    - 현재 정책 πθ_old(a|s)를 고정해 두고
        
    - 여러 환경에서 rollout을 모은다  
        (상태, 행동, 보상, 다음 상태, done 플래그 등).
        
2. Advantage 추정
    
    - [[RL 11 - Advantage Function]]과 [[RL 21 - Actor-Critic (구조)]]에서처럼  
        Critic의 V(s) 추정치를 사용해  
        A_t를 계산한다.
        
    - 보통 GAE(Generalized Advantage Estimation) 같은  
        n-step, λ-return 기반의 Advantage를 사용해  
        분산과 바이어스를 적절히 조절한다.
        
3. 정책 비율 r_t 계산
    
    - 수집된 rollout에 대해  
        현재 정책 πθ(a_t | s_t)와  
        데이터 수집 당시 정책 πθ_old(a_t | s_t)를 비교해  
        r_t를 계산한다.
        
4. Clipped objective로 Actor 업데이트
    
    - r_t와 A_t를 이용해  
        “정책이 Advantage 방향으로 바뀌되,  
        너무 과하게 바뀌지 않도록 제한하는” 목적 함수를 만든다.
        
    - 이 목적 함수를 최대화하는 방향으로  
        θ를 여러 epoch에 걸쳐 미니배치 SGD로 업데이트한다.
        
    - 이때 θ_old는 rollout이 끝날 때까지 고정이고,  
        새 데이터 수집 시점에만 θ_old ← θ로 갱신한다.
        

이 과정을 반복하면서  
PPO의 Actor는 안정적으로 좋은 정책을 향해 나아가게 된다.

---

### 3. Critic 및 Actor-Critic 구조

PPO는 기본적으로 Actor-Critic 구조 위에 올라간다.

- Actor
    
    - 정책 πθ(a|s)를 출력하는 네트워크
        
    - 위에서 설명한 clipped objective + Advantage를 사용해 업데이트
        
- Critic
    
    - 상태 가치 Vϕ(s)를 출력하는 네트워크
        
    - Return 또는 TD 타깃과의 제곱 오차(Value loss)를 최소화하도록 학습
        

실제 손실 함수는 보통 세 가지 항목으로 구성된다.

1. Policy loss
    
    - 클리핑이 들어간 surrogate objective의 음수를  
        최소화하는 형태로 구현된다.
        
2. Value loss
    
    - Vϕ(s_t)가 목표값(예: n-step Return, λ-return)에  
        가까워지도록 제곱 오차를 줄이는 손실이다.
        
3. Entropy bonus
    
    - 정책의 엔트로피를 높이는 보너스를 추가해  
        탐색을 유지하고  
        정책이 지나치게 결정론적으로 수렴하는 것을 방지한다.
        

최종 손실은  
Policy loss + c1 · Value loss − c2 · Entropy term  
형태의 가중합으로 구성되는 경우가 많다.

---

### 4. 설계 및 실무 관점 요약

PPO를 설계·튜닝할 때 자주 고려하는 포인트들이다.

- 클리핑 범위 ε
    
    - ε가 너무 작으면  
        정책이 거의 변하지 않아 학습이 느려지고
        
    - ε가 너무 크면  
        PPO의 장점(정책 변화 제한)이 약해진다.
        
- rollout 길이, 배치 크기
    
    - 한 번에 얼마나 긴 trajectory를 모아서  
        몇 번의 epoch로 학습할지  
        환경 특성과 함께 튜닝해야 한다.
        
- Advantage 추정 방식
    
    - GAE(λ)를 사용할지,
        
    - 단순 n-step Return을 사용할지에 따라  
        안정성과 속도가 달라진다.
        
- Actor와 Critic의 균형
    
    - Value loss 가중치,
        
    - Entropy 보너스 가중치를 어떻게 둘지에 따라  
        탐색–수렴 간 트레이드오프가 달라진다.
        

큰 그림에서 보면

- PPO는 [[RL 20 - Policy Gradient (기본 PG)]]와 [[RL 21 - Actor-Critic (구조)]]에  
    “정책 변화량을 제한하는 클리핑 목적 함수”를 더한 알고리즘이고,
    
- 복잡한 환경에서도 비교적 안정적으로 잘 동작해  
    게임, 로봇, RLHF 같은 대규모 응용에서  
    기본 선택지로 널리 사용되는 현대 표준 Actor-Critic 계열 방법이다.