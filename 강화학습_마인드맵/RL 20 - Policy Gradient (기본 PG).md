상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 01 - Reinforcement Learning]], [[RL 02 - Agent]], [[RL 09 - Policy (정책)]], [[RL 10 - Value Function (V, Q)]], [[RL 11 - Advantage Function]], [[RL 21 - Actor-Critic (구조)]], [[RL 22 - Advantage 기반 A2C·A3C]], [[RL 23 - Proximal Policy Optimization (PPO)]]

---

## What (정의)

Policy Gradient 방법은  
정책 π를 확률모형 π_θ(a|s)로 두고,  
“파라미터 θ를 어느 방향으로 바꿔야 기대 Return이 커지는가”를  
직접 추정해서 θ를 업데이트하는 강화학습 기법이다.

핵심 아이디어를 정리하면 다음과 같다.

- 정책 π_θ(a|s)는  
    상태 s가 들어오면 행동 a에 대한 확률 분포를 내는 함수다.
    
- 우리가 최대화하고 싶은 목표는  
    이 정책을 따라 에피소드를 실행했을 때의  
    기대 Return J(θ)이다.
    
- Policy Gradient는  
    J(θ)를 직접 미분한 ∇_θ J(θ)의 추정값을 구해서  
    θ ← θ + α ∇_θ J(θ)  
    형태로 gradient ascent를 수행한다.
    

즉, “Return이 좋아지는 방향으로  
직접 정책 파라미터를 밀어 올리는” 방법이라고 이해할 수 있다.

---

## Why (배경/목적)

정책 기반 방법이 필요한 이유는  
Value-based 방법만으로는 다루기 까다로운 상황들이 많기 때문이다.

정책 Gradient가 중요한 이유를 정리하면 다음과 같다.

1. 연속 행동공간에 자연스럽게 적용 가능
    
    - [[RL 15 - Q-learning]], [[RL 17 - Deep Q-Network (DQN)]] 같은  
        Value-based 방법은 기본적으로 이산 행동공간에 맞춰져 있다.
        
    - 연속 제어(로봇 토크, 조향각, 속도 등)에서는  
        max_a Q(s, a)를 푸는 것이 어렵거나  
        비현실적일 수 있다.
        
    - Policy Gradient에서는  
        애초에 π_θ(a|s)를 연속 분포(가우시안 등)로 두고  
        그 분포의 파라미터를 직접 최적화하면 된다.
        
2. 정책을 직접 최적화
    
    - Value-based 방법은  
        먼저 Q(s, a)를 학습한 뒤  
        그 위에서 정책을 유도한다.
        
    - Policy Gradient는  
        “최종적으로 필요한 것 = 정책”이라는 관점에서  
        J(θ)를 직접 최적화하기 때문에  
        구조가 직관적이다.
        
3. 확률적 정책을 자연스럽게 다룸
    
    - 복잡한 탐색 전략이나,  
        확률적으로 행동해야 하는 문제를  
        바로 정책 수준에서 모델링할 수 있다.
        
    - 예: 여러 행동을 섞어서 써야 하는 게임 전략 등.
        
4. Actor-Critic, PPO 등 현대 알고리즘의 출발점
    
    - Policy Gradient의 기본 형태(REINFORCE)를  
        개선한 것이  
        [[RL 21 - Actor-Critic (구조)]], [[RL 22 - Advantage 기반 A2C·A3C]], [[RL 23 - Proximal Policy Optimization (PPO)]] 등이다.
        
    - 이들을 이해하기 위한 기본 골격이 바로 Policy Gradient다.
        

요약하면,  
정책 Gradient는 “정책 자체를 직접 다루면서,  
연속 행동·복잡 전략 환경까지 포괄하기 위한 기본 도구”라고 볼 수 있다.

---

## How (활용)

### 1. 정책 표현 방식

Policy Gradient에서는 정책 π_θ(a|s)를  
명시적으로 파라미터화한다.

- 이산 행동 공간
    
    - 신경망이 상태 s를 입력받아  
        각 행동에 대한 로짓 또는 확률을 출력한다.
        
    - 예: softmax를 통해 행동 확률 벡터 π_θ(a|s)를 만든 뒤,  
        그 분포에서 행동을 샘플링한다.
        
- 연속 행동 공간
    
    - 신경망이 상태 s를 입력받아  
        가우시안 분포의 평균, 분산 등을 출력한다.
        
    - 행동은 a ~ N(μ_θ(s), Σ_θ(s)) 같은 방식으로 샘플링한다.
        

이 정책 파라미터 θ를  
“Return이 증가하는 방향”으로 업데이트하는 것이  
정책 Gradient의 목표다.

---

### 2. 기본 REINFORCE 알고리즘 개념

가장 기본적인 Policy Gradient 알고리즘이  
REINFORCE로 불리는 Monte Carlo Policy Gradient이다.

개념적인 흐름은 다음과 같다.

1. 정책 π_θ로 에피소드 생성
    
    - 현재 파라미터 θ로 정의된 정책에 따라  
        하나의 에피소드를 끝까지 진행한다.
        
    - (s_0, a_0, r_1, s_1, a_1, r_2, …, s_T)  
        형태의 trajectory를 얻는다.
        
2. 각 시점 t에 대해 Return G_t 계산
    
    - 시점 t 이후로 받는 (할인된) 보상의 누적합을  
        G_t로 계산한다.
        
    - G_t는 “그 시점에서 본다면 얼마나 잘 됐는지”를 나타내는 척도다.
        
3. Policy Gradient 업데이트
    
    - 각 시점 t에 대해  
        행동 a_t가 실제로 선택된 로그 확률 log π_θ(a_t | s_t)에  
        해당 시점의 Return G_t를 곱해  
        그 합을 gradient ascent 방향으로 올린다.
        
    - 직관적으로는
        
        - G_t가 큰 행동일수록  
            그 행동이 나올 확률을 크게 늘리고
            
        - G_t가 작은 행동일수록  
            그 행동이 나올 확률을 줄이도록  
            θ를 조정하는 과정이다.
            
4. 여러 에피소드를 통해 반복
    
    - 위 과정을 반복하면서  
        평균 Return이 점점 커지는 방향으로  
        θ를 학습해 간다.
        

REINFORCE는 구조가 매우 단순하고  
정책 Gradient의 핵심 아이디어를 잘 보여 주지만,

- Return 자체의 분산이 크고
    
- 학습이 느리며 불안정할 수 있다는 단점이 있다.
    

이 문제를 해결하기 위해  
Baseline을 빼거나([[RL 11 - Advantage Function]]),  
Value Function을 함께 학습하는  
Actor-Critic 구조 등이 등장한다.

---

### 3. Baseline, Advantage와의 연결

기본 REINFORCE에서는 G_t를 그대로 쓰지만,  
실전에서는 다음과 같은 개선을 많이 사용한다.

- Baseline b(s) 도입
    
    - G_t 대신 (G_t − b(s_t))를 사용해  
        gradient의 분산을 줄인다.
        
    - b(s)는 보통 상태 가치 V(s)의 추정치가 된다.
        
- Advantage A(s, a) 사용
    
    - [[RL 11 - Advantage Function]]에서 보는 것처럼  
        Q(s, a) − V(s) 또는  
        (G_t − V(s_t)) 형태의 Advantage를 쓰면  
        “평균보다 얼마나 더 나았는지” 기준으로  
        정책을 업데이트하게 된다.
        
    - 이는 [[RL 21 - Actor-Critic (구조)]], [[RL 22 - Advantage 기반 A2C·A3C]]의 기본 아이디어다.
        

이렇게 Baseline, Advantage를 도입하면

- Policy Gradient의 본질은 유지하면서도
    
- 학습 신호의 분산이 크게 줄고
    
- 수렴 속도와 안정성이 개선된다.
    

---

### 4. Actor-Critic, PPO로의 확장

기본 Policy Gradient는  
직접적으로는 REINFORCE 형식이지만,  
현대 RL에서는 거의 항상 개선된 형태를 사용한다.

- Actor-Critic 구조([[RL 21 - Actor-Critic (구조)])
    
    - Actor: 정책 π_θ(a|s)를 학습
        
    - Critic: 가치 함수 V_w(s) 또는 Q_w(s, a)를 학습
        
    - Critic이 제공하는 가치/Advantage를  
        Actor의 Policy Gradient에 곱해  
        분산을 줄이고 안정적으로 학습한다.
        
- Advantage 기반 A2C·A3C([[RL 22 - Advantage 기반 A2C·A3C]])
    
    - 여러 워커가 병렬로 trajectory를 수집하고
        
    - Advantage를 이용해 정책을 업데이트하는 구조
        
    - 기본 골격은 여전히 Policy Gradient다.
        
- PPO([[RL 23 - Proximal Policy Optimization (PPO)])
    
    - Policy Gradient 업데이트 폭을  
        클리핑된 objective로 제한해  
        너무 큰 정책 변화로 인한 불안정을 줄인다.
        
    - 기본적으로는 “옛 정책과 새 정책의 비율 × Advantage”를  
        최적화하는 Policy Gradient의 개선 버전이다.
        

이 모든 알고리즘의 공통점은

- “정책 π_θ를 직접 최적화한다”는  
    Policy Gradient의 철학을 공유한다는 점이다.
    

---

### 5. 설계 및 실무 관점 요약

- Policy Gradient의 핵심 한 줄 정리
    
    - 정책을 확률모형 π_θ로 두고,  
        기대 Return J(θ)를 직접 최대화하는 방향으로  
        θ를 업데이트하는 방법이다.
        
- 언제 유리한가
    
    - 연속 행동공간
        
    - 복잡한 전략, 확률적 정책이 중요한 문제
        
    - Q를 argmax로 풀기 번거로운 환경
        
- 순수 REINFORCE의 한계
    
    - Return의 분산이 크고,  
        에피소드 길이가 길면 학습이 느리다.
        
    - 그래서 실전에서는 거의 항상  
        Baseline, Advantage, Actor-Critic, PPO 같은  
        개선 기법과 함께 사용된다.
        
- 큰 그림에서의 위치
    
    - Value-based RL이 “V, Q를 먼저 학습해서 정책을 유도”한다면,
        
    - Policy-based RL(Policy Gradient)은  
        “정책을 직접 최적화”하는 축이다.
        
    - 이 두 축과 [[RL 21 - Actor-Critic (구조)]]를 함께 이해하면  
        대부분의 현대 RL 알고리즘을  
        큰 틀에서 분류하고 비교하기 쉬워진다.