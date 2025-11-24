상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 01 - Reinforcement Learning]], [[RL 02 - Agent]], [[RL 04 - State]], [[RL 09 - Policy (정책)]], [[RL 10 - Value Function (V, Q)]], [[RL 15 - Q-learning]], [[RL 17 - Deep Q-Network (DQN)]], [[RL 20 - Policy Gradient (기본 PG)]], [[RL 23 - Proximal Policy Optimization (PPO)]]

---

## What (정의)

Action은  
Agent([[RL 02 - Agent]])가 현재 상태에서 선택하는 실제 행동이다.

Action Space는  
Agent가 선택할 수 있는 모든 행동의 집합으로,  
이 집합의 구조와 형태에 따라  
가능한 정책과 알고리즘이 달라진다.

강화학습에서 보통 다음처럼 정리한다.

- 시점 t에서 Agent는 상태 s_t를 보고  
    Action Space A에서 행동 a_t ∈ A 하나를 선택한다.
    
- 정책 π(a|s)는  
    상태 s에서 각 행동 a를 선택할 확률(또는 규칙)을 정의한다.
    
- Action Space는 크게  
    이산(discrete)과 연속(continuous)으로 나눈다.
    

요약하면, Action은 “지금 할 일”이고  
Action Space는 “선택 가능한 모든 일들의 목록”이다.

---

## Why (배경/목적)

Action Space의 형태는  
선택 가능한 알고리즘과 정책 구조를 직접적으로 제한한다.

- 이산형 Action Space
    
    - 행동의 개수가 명확히 셀 수 있을 때  
        예: 왼/오/점프/공격, 메뉴에서 N개 중 하나 선택
        
    - Q-learning, SARSA, DQN처럼  
        각 행동에 대한 Q(s, a)를 직접 추정하는 알고리즘이 잘 맞는다.
        
- 연속형 Action Space
    
    - 행동이 연속 값일 때  
        예: 조향각, 가속도, 힘의 크기, 연속적인 비율 등
        
    - 모든 가능한 실수값 a에 대해  
        Q(s, a)를 직접 테이블처럼 관리하는 것은 불가능하다.
        
    - 그래서 Policy Gradient, Actor-Critic, PPO, DDPG 등  
        정책을 직접 파라미터화하는 방법이 주로 사용된다.
        

또한 Action Space를 어떻게 정의하느냐에 따라

- 탐색 난이도
    
- 학습 안정성
    
- 제약 조건(안전, 물리 한계 등)을 표현하는 방식
    

이 모두가 영향을 받는다.

즉, Action Space 설계는  
“에이전트가 무엇을 할 수 있는지”와  
“어떤 알고리즘을 쓸 수 있는지”를 동시에 결정하는 핵심 요소다.

---

## How (활용)

### 1. Action Space의 유형

1. 이산형 Action Space
    
    - 예:
        
        - 게임: 왼, 오른쪽, 점프, 공격
            
        - 메뉴 선택: 상품 A/B/C 중 하나
            
    - 보통 행동을 0, 1, 2, …, N-1 같은 인덱스로 표현한다.
        
    - Q-learning, DQN에서는  
        상태를 입력으로 받아 각 행동에 대한 Q값 벡터를 출력하고,  
        argmax를 취해 행동을 선택한다.
        
    - 탐색은 ε-greedy처럼 일부 확률로 랜덤 행동을 섞는 방식이 자주 사용된다.
        
2. 연속형 Action Space
    
    - 예:
        
        - 자율주행: 조향각, 가속도
            
        - 로봇 제어: 관절 토크, 속도 명령
            
    - 행동 a가 실수 벡터가 되며,  
        정책 네트워크는 보통 가우시안 분포의 평균과 분산 등을 출력한다.
        
    - Policy Gradient, PPO, SAC 같은 알고리즘에서  
        확률적 정책 π(a|s)를 직접 파라미터화하여 학습한다.
        
3. 혼합형 / 다차원 Action Space
    
    - 일부 컴포넌트는 이산, 일부는 연속일 수 있다.
        
    - 예:
        
        - “어떤 모드를 선택할지”(이산) + “그 모드에서의 강도 값”(연속)
            
    - 구현에서는 여러 헤드(head)를 가진 정책 네트워크를 사용하거나,  
        구조를 나누어 처리한다.
        

---

### 2. Gym 스타일에서의 Action Space

OpenAI Gym 스타일 환경에서는  
Action Space가 env.action_space로 정의된다.

- 이산형:
    
    - 예: `Discrete(n)`
        
    - 가능한 행동은 0, 1, ..., n-1
        
- 연속형:
    
    - 예: `Box(low, high, shape)`
        
    - 각 차원이 주어진 범위 내의 실수 값을 가진다.
        

Agent의 정책 네트워크나 행동 선택 로직은  
항상 이 env.action_space의 정의를 만족하는 형태로  
행동을 생성해야 한다.

---

### 3. Policy 출력과 Action Space의 관계

정책 네트워크는  
Action Space 구조에 맞는 출력 형태를 가져야 한다.

- 이산형 Action Space
    
    - 상태를 입력으로 받아  
        각 행동에 대한 로짓(logit) 또는 확률 벡터를 출력한다.
        
    - 마지막 층에서 softmax를 적용해  
        행동 확률 분포 π(a|s)를 만든 뒤,  
        그 중 하나를 샘플링하거나 argmax로 선택한다.
        
- 연속형 Action Space
    
    - 상태를 입력으로 받아  
        각 Action 차원에 대한 평균(μ)과 분산(또는 로그 분산)을 출력한다.
        
    - 이 값들로 가우시안 분포 등을 구성하고,  
        거기서 샘플링한 a를 행동으로 사용한다.
        
    - Proximal Policy Optimization(PPO), Policy Gradient 계열에서는  
        이 확률분포의 log π(a|s)를 이용해  
        gradient를 계산한다.
        
- 결정적 정책
    
    - 일부 알고리즘(예: DDPG, TD3)에서는  
        확률 분포가 아니라  
        상태에서 곧바로 행동 벡터 a를 출력하는  
        결정적 정책을 사용한다.
        
    - 이때는 행동에 노이즈를 추가해 탐색을 수행한다.
        

---

### 4. 설계 및 실무 관점 포인트

- Action Space를 가능한 한 단순하게 설계하기
    
    - 불필요한 행동을 줄이면  
        탐색 난이도가 크게 낮아진다.
        
    - 예를 들어,  
        “왼 1칸, 왼 2칸, 왼 3칸 …”처럼  
        세분화된 액션보다  
        “왼쪽, 오른쪽, 점프”처럼  
        최소한의 의미 있는 행동만 두는 편이  
        학습이 쉬울 수 있다.
        
- 물리적/시스템적 제약 반영하기
    
    - 로봇 토크, 속도, 조향각에는 안전/물리 한계가 있으므로  
        Action Space에 이 범위를 명시해 두면  
        정책이 위험한 행동을 내지 않도록 자연스럽게 제한할 수 있다.
        
- 탐색 전략과의 연계
    
    - 이산형에서는 ε-greedy, 볼츠만 탐색 등  
        단순한 전략으로도 어느 정도 동작한다.
        
    - 연속형에서는 행동에 가우시안 노이즈를 추가하거나  
        정책 자체를 확률분포로 모델링해야 한다.
        
- 알고리즘 선택 가이드
    
    - 이산형: Q-learning, DQN, Double DQN, Dueling DQN 등
        
    - 연속형: Policy Gradient, Actor-Critic, PPO, DDPG, TD3, SAC 등
        
    - 복잡한 연속 제어 문제는  
        처음부터 Q-learning 계열보다는  
        Policy Gradient/Actor-Critic 계열을 우선 고려하는 것이 보통 더 자연스럽다.