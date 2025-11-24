상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 01 - Reinforcement Learning]], [[RL 02 - Agent]], [[RL 03 - Environment & MDP]], [[RL 04 - State]], [[RL 05 - Action & Action Space]], [[RL 07 - Return & Discount Factor γ]], [[RL 10 - Value Function (V, Q)]], [[RL 20 - Policy Gradient (기본 PG)]], [[RL 21 - Actor-Critic (구조)]], [[RL 23 - Proximal Policy Optimization (PPO)]]

---

## What (정의)

Policy(정책)는  
상태를 입력받아 어떤 행동을 할지 결정하는 규칙 또는 함수다.

- 시점 t에서 상태 s_t를 보면  
    정책 π가 행동 a_t를 정해 준다.
    
- 확률적 정책(stochastic policy)
    
    - π(a|s)는 상태 s에서 행동 a를 선택할 확률 분포
        
    - 같은 상태 s에서도 매번 다른 행동이 나올 수 있다.
        
- 결정론적 정책(deterministic policy)
    
    - μ(s) = a 형태로 항상 같은 행동을 내는 함수
        
    - 같은 상태 s면 항상 동일한 a를 선택한다.
        

강화학습에서 Policy는  
에이전트([[RL 02 - Agent]])가 세계([[RL 03 - Environment & MDP]])와 상호작용할 때  
실제로 행동을 만들어 내는 핵심 객체라고 볼 수 있다.

요약하면, Policy는  
어떤 상황(State)에서 어떤 행동(Action)을 할지에 대한  
에이전트의 행동 전략 전체를 담은 함수다.

---

## Why (배경/목적)

강화학습의 궁극적인 산출물은  
좋은 값 함수 하나가 아니라,  
좋은 Policy다.

- 현실 문제에서 궁금한 것
    
    - “이 시스템은 어떤 상황에서 어떻게 행동해야 하는가?”
        
    - 이 질문에 대한 답이 바로 Policy다.
        
- Value Function(V, Q)은
    
    - 특정 상태나 상태–행동 쌍이  
        앞으로 얼마만큼의 Return을 가져올지 평가하는 도구이고
        
    - Policy는 실제로 행동을 선택하는 규칙이다.
        

그래서 많은 알고리즘의 최종 목표는

- 어떤 기준(예: 기대 Return)을 최대화하는  
    최적 정책 π*를 찾는 것
    
- 이때 Value Function은  
    정책을 평가하고 개선하는 중간 도구 역할을 하는 경우가 많다.
    

또한 Policy는 다음과 같은 측면에서 중요하다.

- 탐색 전략을 내장할 수 있다.
    
    - 확률적 정책은 자연스럽게 여러 행동을 시도하게 만들어  
        탐색을 유도할 수 있다.
        
- 제약 조건을 직접 반영하기 좋다.
    
    - 특정 행동을 절대 하지 않도록  
        Policy 구조에 제약을 넣을 수 있다.
        
- 실제 시스템 적용 시
    
    - “현재 상태 들어오면 바로 행동이 나가는 함수” 형태라  
        배포·서빙하기에 적합하다.
        

---

## How (활용)

### 1. Value-based 접근에서의 Policy

Value-based 방식은  
먼저 행동 가치 Q(s, a)를 학습한 뒤,  
그 Q로부터 Policy를 유도한다.

대표 예시: [[RL 15 - Q-learning]], [[RL 17 - Deep Q-Network (DQN)]]

- 기본 greedy 정책
    
    - 상태 s에서 Q(s, a)가 최대인 행동을 선택
        
        - a = argmax_a Q(s, a)
            
    - 현재 추정 상으로 가장 좋은 행동만 택하는 전략이다.
        
- ε-greedy 정책
    
    - 대부분의 경우 greedy 행동을 선택하되  
        일정 확률 ε로 랜덤 행동을 선택해 탐색
        
        - 확률 1−ε: argmax_a Q(s, a)
            
        - 확률 ε: 무작위 행동
            
    - 학습 초반에는 ε를 크게,  
        나중에는 점점 줄이는 스케줄을 쓰는 경우가 많다.
        
- softmax / Boltzmann 정책
    
    - Q(s, a)를 온도 파라미터 τ로 스케일링해 softmax로 확률 분포를 만든 뒤  
        그 분포에서 행동을 샘플링
        
    - Q가 큰 행동은 자주, 작은 행동도 가끔은 선택하도록 만드는 방식
        

이처럼 Value-based 알고리즘에서 Policy는  
Q 함수를 기반으로 정의되는 파생물에 가깝다.

---

### 2. Policy-based 접근에서의 Policy

Policy-based 방식은  
정책 자체를 파라미터 θ를 가진 함수로 두고  
직접 학습한다.

대표 예시: [[RL 20 - Policy Gradient (기본 PG)]], [[RL 23 - Proximal Policy Optimization (PPO)]], [[RL 21 - Actor-Critic (구조)]]

- 확률적 정책 π_θ(a|s)
    
    - 신경망이 상태 s를 입력받아  
        각 행동에 대한 확률 분포를 출력
        
        - 이산 행동 공간:
            
            - 로짓 → softmax → 행동 확률
                
        - 연속 행동 공간:
            
            - 평균, 분산 → 가우시안 분포 → 샘플링
                
- 결정론적 정책 μ_θ(s)
    
    - 상태를 입력받아  
        바로 연속형 행동 벡터를 출력
        
    - DDPG, TD3 같은 연속 제어 알고리즘에서 자주 사용된다.
        

Policy-based 방법에서는

- 목적 함수 J(θ) = 기대 Return
    
- 이 J(θ)를 gradient ascent로 최대화하도록  
    ∇_θ J(θ)를 추정해 파라미터 θ를 업데이트한다.
    

즉, Policy 그 자체가  
최적화의 직접적인 대상이 된다.

---

### 3. Policy와 탐색(exploration)

Policy 설계는 탐색 전략과도 밀접하게 연결된다.

- 확률적 정책
    
    - 같은 상태에서 다른 행동이 나올 수 있어  
        탐색이 자연스럽게 일어난다.
        
    - Policy Gradient 계열에서 기본적으로 사용하는 형태다.
        
- 결정론적 정책 + 노이즈
    
    - 행동에 가우시안 노이즈 등을 추가해  
        연속 공간에서 탐색을 수행할 수 있다.
        
    - 예: a = μ_θ(s) + noise
        
- ε-greedy 정책
    
    - 이산 행동에서 가장 단순하고 널리 쓰이는 탐색 방식
        
    - ε를 서서히 줄여가며  
        초반에는 탐색, 후반에는 exploitation에 더 집중하도록 조정할 수 있다.
        

탐색과 Policy는 분리된 개념이지만,  
실제로 구현할 때는 Policy 정의 안에  
탐색 전략을 녹여서 함께 설계하는 경우가 많다.

---

### 4. Policy와 Value Function의 관계

Policy와 Value Function은  
서로를 보완하는 역할을 한다.

- 주어진 Policy π에 대해
    
    - V^π(s), Q^π(s, a)는  
        그 정책이 얼마나 좋은지 평가하는 기준
        
- Value-based RL
    
    - Q를 통해 Policy를 유도
        
    - Policy는 Q의 argmax(또는 ε-greedy 등)로 파생
        
- Policy-based RL
    
    - Policy를 직접 최적화
        
    - Value Function은
        
        - Return의 분산을 줄이고
            
        - 학습을 안정화하기 위한 baseline, Advantage 계산에 사용
            

Actor-Critic 구조([[RL 21 - Actor-Critic (구조)]])는

- Actor: Policy(행동 전략)를 담당하는 네트워크
    
- Critic: Value Function(평가자)을 담당하는 네트워크
    

이 둘을 동시에 학습하여  
Policy와 Value의 장점을 모두 활용하려는 접근이다.

---

### 5. 설계 및 실무 관점 요약

- 결국 배포되는 것은 Policy다
    
    - 실제 서비스/로봇/게임에 넣어 돌아가는 것은  
        상태를 입력받아 즉시 행동을 내는 π 또는 μ이다.
        
- Value-based로 시작할지, Policy-based로 시작할지
    
    - 이산 행동, 비교적 단순한 환경 → Value-based(DQN 계열)도 적합
        
    - 연속 제어, 복잡한 전략, 탐색 구조가 중요한 문제 → Policy-based, Actor-Critic 계열 고려
        
- 탐색과 제약을 어떻게 정책에 녹일지
    
    - 안전 제약, 행동 범위, 특정 행동 금지 등을  
        Policy 설계에서 직접 반영할 수 있다.
        
- 실무 코드 구조
    
    - 보통 Agent 안에
        
        - policy(state) → action
            
        - value(state), q_value(state, action)  
            같은 메서드를 두고,  
            학습 루프에서는  
            이 둘을 어떻게 업데이트할지에 따라  
            알고리즘이 달라진다고 정리할 수 있다.