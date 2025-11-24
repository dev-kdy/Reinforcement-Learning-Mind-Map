상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 01 - Reinforcement Learning]], [[RL 06 - Reward]], [[RL 07 - Return & Discount Factor γ]], [[RL 09 - Policy (정책)]], [[RL 11 - Advantage Function]], [[RL 13 - Monte Carlo Learning]], [[RL 14 - Temporal-Difference (TD) Learning]], [[RL 15 - Q-learning]], [[RL 17 - Deep Q-Network (DQN)]]

---

## What (정의)

Value Function은  
어떤 상태 또는 상태–행동 쌍이  
앞으로 받을 누적 보상(Return) 관점에서  
얼마나 좋은지를 나타내는 함수다.

정책 π에 대해 보통 두 가지 형태를 쓴다.

- 상태 가치 함수 V^π(s)
    
    - 상태 s에서 시작해,  
        그 이후 정책 π를 따라 행동했을 때  
        기대되는 Return의 크기를 나타낸다.
        
    - 직관적으로 “지금 이 상태에 있는 것이 얼마나 좋은가”를 점수로 표현한 것.
        
- 행동 가치 함수 Q^π(s, a)
    
    - 상태 s에서 행동 a를 먼저 한 번 선택한 뒤,  
        그 이후 정책 π를 따라 행동했을 때  
        기대되는 Return의 크기를 나타낸다.
        
    - 직관적으로 “지금 이 상태에서 이 행동을 하는 것이 얼마나 좋은가”에 대한 점수다.
        

강화학습에서 Value Function은  
정책의 장기 성능을 수치적으로 평가하는 핵심 도구다.

---

## Why (배경/목적)

직접 정책을 바로 바꾸는 것보다,  
먼저 각 상태와 행동의 “좋고 나쁨”을 수치로 정의해 두면  
정책을 체계적으로 개선하기가 훨씬 쉽다.

Value Function이 중요한 이유를 정리하면 다음과 같다.

1. 정책 평가의 기준이 된다
    
    - 특정 정책 π가 얼마나 좋은 정책인지  
        단순히 “감”이 아니라  
        V^π(s), Q^π(s, a) 같은 수치로 평가할 수 있다.
        
    - 이 기준 위에서  
        “현재 정책보다 더 나은 정책”을 설계할 수 있다.
        
2. Value-based RL의 기반이 된다
    
    - [[RL 15 - Q-learning]], [[RL 17 - Deep Q-Network (DQN)]]처럼  
        Q(s, a)만 잘 학습해 두면  
        그 위에서 greedy, ε-greedy 등으로  
        좋은 정책을 뽑아낼 수 있다.
        
    - 이때 정책은 Value Function에서 파생되는 존재가 된다.
        
3. Bellman 식과 동적 계획법(DP)을 가능하게 한다
    
    - Value Function이 있으면  
        “현재 가치 = 즉시 보상 + 다음 상태 가치”라는  
        재귀적인 관계를 만들 수 있다.
        
    - 이것이 벨만 방정식이고,  
        value iteration, policy iteration 같은  
        동적 계획법 알고리즘의 기반이 된다.
        
4. Policy Gradient에서 분산을 줄이는 역할
    
    - [[RL 11 - Advantage Function]]에서처럼  
        Q와 V를 이용해 “평균보다 얼마나 좋은가”를 계산해  
        정책 gradient의 분산을 줄이고  
        더 안정적으로 학습할 수 있다.
        

요약하면, Value Function은  
“정책을 직접 바꾸기 전에,  
상태/행동의 장기적인 품질을 수치화해 주는 중간 레이어” 역할을 한다.

---

## How (활용)

### 1. 가치 함수의 종류와 관계

Value Function은 보통 다음 두 가지에 집중해 쓴다.

1. 상태 가치 함수 V(s)
    
    - 어떤 정책 π가 고정되어 있을 때,  
        상태 s에서 시작해서 장기적으로 얼마나 많은 Return을 기대할 수 있는지.
        
    - 상태만 보고 의사결정해야 할 때  
        “이 상황이 전반적으로 좋은지 나쁜지”를 판단하는 기준이 된다.
        
2. 행동 가치 함수 Q(s, a)
    
    - 상태 s에서 특정 행동 a를 선택했을 때,  
        그 선택이 장기적으로 얼마나 이득인지 평가한다.
        
    - greedy 정책
        
        - a = argmax_a Q(s, a)  
            형태로 정책을 쉽게 뽑아낼 수 있어서  
            Value-based RL의 핵심 도구가 된다.
            

이 두 함수는 [[RL 11 - Advantage Function]]에서  
A(s, a) = Q(s, a) − V(s) 같은 형태로 결합되어  
“평균적인 상태 가치보다 얼마나 더 좋은 행동인가”를 나타내는 데에도 사용된다.

---

### 2. 학습 방법: Monte Carlo vs TD

Value Function V, Q는  
Return을 직접 계산해서 추정할 수도 있고,  
현재 추정치를 이용해 점진적으로 보정할 수도 있다.

1. Monte Carlo 방식([[RL 13 - Monte Carlo Learning]])
    
    - 한 에피소드가 끝나기를 기다린 뒤,  
        실제로 받은 보상들을 모두 합쳐 Return G_t를 구하고  
        그 값을 이용해 V(s) 또는 Q(s, a)를 업데이트한다.
        
    - 장점
        
        - 에피소드 기준으로 보면,  
            편향이 없는 정확한 Return을 사용한다.
            
    - 단점
        
        - 에피소드가 끝날 때까지 기다려야 업데이트가 가능하고  
            긴 에피소드에서는 학습이 느려지고 분산이 커질 수 있다.
            
2. Temporal-Difference(TD) 방식([[RL 14 - Temporal-Difference (TD) Learning]])
    
    - 한 스텝씩 진행하면서  
        실제 Reward와 다음 상태의 가치 추정치를 섞어 업데이트한다.
        
    - 예시 느낌
        
        - V(s_t) ≈ r_{t+1} + γ V(s_{t+1})
            
    - 장점
        
        - 한 스텝마다 바로 업데이트 가능해서  
            온라인 학습, 실시간 환경에 적합하다.
            
        - DP(동적 계획법)와 Monte Carlo의 장점을 절충한 형태라고 볼 수 있다.
            
    - 단점
        
        - 다음 상태 가치 V(s_{t+1}) 추정치에 의존하므로  
            어느 정도의 편향이 존재할 수 있다.
            

실전 알고리즘에서는  
n-step Return, λ-return처럼  
Monte Carlo와 TD 사이를 조절하는 변형도 많이 사용한다.

---

### 3. Q-learning, DQN과 Value Function

[[RL 15 - Q-learning]]과 [[RL 17 - Deep Q-Network (DQN)]]은  
Value Function, 특히 Q(s, a)를 중심에 두는 대표적인 알고리즘이다.

- Q-learning
    
    - Q(s, a)를 업데이트할 때  
        실제 Reward와 다음 상태의 최대 Q값을 사용한다.
        
        - 대략적인 형태
            
            - Q(s, a) ← Q(s, a) + α [ r + γ max_{a'} Q(s', a') − Q(s, a) ]
                
    - 여기서 r + γ max_{a'} Q(s', a')가  
        “새로운 타깃 가치 추정치” 역할을 한다.
        
- DQN
    
    - Q(s, a)를 신경망으로 근사하는 버전
        
    - 상태를 입력으로 넣으면  
        가능한 모든 행동에 대한 Q값 벡터가 나오고,  
        그 중 최대 Q값을 가지는 행동을 선택한다.
        
    - 경험 재플레이, 타깃 네트워크 등을 이용해  
        안정적으로 Q 함수를 학습한다.
        

이처럼 Value Function을 잘 학습하기만 하면  
정책은 Q로부터 쉽게 뽑을 수 있다는 점이  
Value-based RL의 큰 장점이다.

---

### 4. Policy Gradient, Actor-Critic에서의 역할

Policy를 직접 학습하는  
Policy-based, Actor-Critic 계열에서도  
Value Function은 여전히 중요한 조연이다.

- Policy Gradient 기본 형태에서는  
    Return만으로 gradient를 계산할 수 있지만  
    분산이 매우 커지는 문제가 있다.
    
- 이를 완화하기 위해  
    상태 가치 V(s) 또는 Advantage A(s, a)를 사용한다.
    
    - 예:
        
        - “평균적으로 이 상태에서 받는 Return보다  
            이 행동이 얼마나 더 좋았는지”를 기준으로  
            정책 파라미터를 업데이트하면  
            잡음이 줄어들고 학습이 안정된다.
            

Actor-Critic 구조에서

- Critic은 V(s) 또는 Q(s, a)를 근사하는 네트워크
    
- Actor는 Policy π(a|s)를 근사하는 네트워크
    

이 둘을 함께 학습함으로써  
Value Function과 Policy를 동시에 잘 맞춰 가는 것이 목표다.

---

### 5. 설계 및 실무 관점 요약

- Value Function은  
    “장기적인 좋고 나쁨”을 수치로 표현한 지표다.
    
- Value-based RL
    
    - Q(s, a)를 잘 학습하고
        
    - 그 위에서 greedy/ε-greedy 정책을 뽑아 쓰는 구조
        
- Policy-based RL
    
    - Policy를 직접 최적화하지만
        
    - Value Function을 이용해  
        학습 안정성과 효율을 높인다.
        
- 구현할 때는
    
    - `value_net(state)` 또는 `q_net(state, action)` 같은 형태로  
        Value Function을 명시적인 모듈로 두고,
        
    - 업데이트 규칙만 바꿔 주면  
        Monte Carlo, TD, Q-learning, Actor-Critic 등  
        다양한 알고리즘으로 확장할 수 있다.