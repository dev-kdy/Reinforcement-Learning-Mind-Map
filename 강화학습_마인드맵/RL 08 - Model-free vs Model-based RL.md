상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 01 - Reinforcement Learning]], [[RL 03 - Environment & MDP]], [[RL 06 - Reward]], [[RL 07 - Return & Discount Factor γ]], [[RL 09 - Policy (정책)]], [[RL 10 - Value Function (V, Q)]], [[RL 13 - Monte Carlo Learning]], [[RL 14 - Temporal-Difference (TD) Learning]], [[RL 15 - Q-learning]], [[RL 17 - Deep Q-Network (DQN)]], [[RL 20 - Policy Gradient (기본 PG)]], [[RL 22 - Policy Gradient]], [[RL 23 - Proximal Policy Optimization (PPO)]]

---

## What (정의)

강화학습에서 Model은  
환경([[RL 03 - Environment & MDP]])의 전이 P(s' | s, a)와 보상 R(s, a)를 의미한다.

이 관점에서 강화학습 알고리즘은 크게 두 가지로 나눌 수 있다.

- Model-free RL
    
    - 환경의 전이 모델 P, 보상 모델 R을 명시적으로 알지 못하거나 사용하지 않는다.
        
    - 대신 샘플 경험 (s, a, r, s')만을 이용해  
        가치 함수([[RL 10 - Value Function (V, Q)]])나  
        정책([[RL 09 - Policy (정책)]])을 직접 학습한다.
        
    - 예: [[RL 15 - Q-learning]], [[RL 17 - Deep Q-Network (DQN)]], [[RL 23 - Proximal Policy Optimization (PPO)]]
        
- Model-based RL
    
    - 환경의 동작, 즉 dynamics(transition model)를  
        알고 있다고 가정하거나, 따로 학습해서 사용한다.
        
    - 이 모델을 활용해 가상의 roll-out, planning을 수행하여  
        정책과 가치 함수를 개선한다.
        
    - 예: value iteration, policy iteration, planning + MPC, Dyna 계열, world model 계열
        

요약하면,  
Model-free는 경험만 보고 바로 정책/가치를 배우는 방식이고,  
Model-based는 환경의 규칙을 먼저 파악한 뒤 그 규칙을 활용해 계획까지 하는 방식이다.

---

## Why (배경/목적)

환경 모델을 활용할 수 있는지, 그리고 그 모델을 얼마나 정확하게 만들 수 있는지는  
현실 문제에서 매우 중요한 설계 포인트다.

- Model-based의 장점
    
    - 환경 모델을 가지고 있으면  
        실제로 환경과 상호작용하지 않고도  
        머릿속(시뮬레이터)에서 여러 행동 시나리오를 비교하는 planning이 가능하다.
        
    - 샘플 하나로부터 여러 가상 roll-out을 생성할 수 있어  
        샘플 효율을 크게 높일 수 있다.
        
    - 제어 이론 쪽에서의 MDP, LQR, MPC 같은 기법은  
        기본적으로 model-based 관점에 속한다.
        
- Model-based의 한계
    
    - 복잡한 현실 환경(로봇, 자율주행, 추천 시스템 등)에서는  
        정확한 전이 모델 P(s' | s, a)를 알기 어렵다.
        
    - 학습한 모델이 조금만 틀려도  
        planning 결과가 크게 왜곡될 수 있다.
        
    - 높은 차원, 복잡한 물리, 노이즈, 비선형성이 클수록  
        모델링 난이도가 매우 상승한다.
        
- Model-free의 장점
    
    - P, R을 직접 모델링하지 않기 때문에  
        모델링 오차를 신경 쓰지 않아도 된다.
        
    - 경험 데이터 (s, a, r, s')만 쌓이면 바로 학습을 진행할 수 있어  
        구현이 상대적으로 단순하다.
        
    - deep RL에서 많이 쓰이는 DQN, PPO, SAC 등은  
        대부분 Model-free 계열이다.
        
- Model-free의 한계
    
    - 같은 성능을 내려면  
        더 많은 실제 환경 상호작용 샘플이 필요할 수 있다.
        
    - 물리 로봇, 실제 시스템 운영처럼  
        데이터 수집이 비싼 환경에서는  
        순수 Model-free만으로는 부담이 크다.
        

그래서 현실에서는

- 환경 모델을 만들 수 있으면 최대한 활용하되
    
- 정확한 모델링이 어려운 부분은 Model-free로 보완하는  
    혼합 접근 방식이 많이 연구되고 있다.
    

---

## How (활용)

### 1. Model-free RL의 전형적인 구조

Model-free는  
환경 모델 없이 샘플 경험만을 이용해  
가치/정책을 직접 업데이트한다.

대표적인 패턴:

1. 경험 수집
    
    - 현재 정책으로 환경을 돌려서  
        상태 s, 행동 a, 보상 r, 다음 상태 s'를 얻는다.
        
2. 학습 대상 정의
    
    - Value-based
        
        - Q(s, a) 또는 V(s)에 대한 타깃을  
            TD, Monte Carlo 방식으로 정의한다.
            
        - 예: Q-learning의 타깃
            
            - r + γ max_{a'} Q(s', a')
                
    - Policy-based
        
        - Return 또는 Advantage를 이용해  
            정책의 gradient를 계산한다.
            
        - 예: Policy Gradient, PPO
            
3. 파라미터 업데이트
    
    - 샘플 (s, a, r, s')를 사용해  
        가치 함수와 정책 네트워크의 파라미터를  
        확률적 경사 하강법으로 업데이트한다.
        
4. 반복
    
    - 환경에서 계속 데이터 수집 → 업데이트를 반복하며  
        점점 정책을 개선한다.
        

이 과정에서 환경은  
그저 step(s, a) → (s', r)를 제공하는 블랙박스일 뿐이고,  
전이 확률 P, 보상 함수 R의 명시적인 수식은 전혀 사용하지 않는다.

---

### 2. Model-based RL의 전형적인 구조

Model-based는  
환경의 전이와 보상에 대한 모델을 활용하는 것이 핵심이다.

크게 두 가지 경우가 있다.

1. 환경 모델을 이미 알고 있는 경우
    
    - 예: 그리드월드, 단순한 MDP, 선형/가우시안 시스템 등
        
    - P(s' | s, a), R(s, a)가 수식이나 테이블로 주어져 있다.
        
    - 이 경우 다음과 같은 planning 알고리즘을 적용할 수 있다.
        
        - value iteration, policy iteration
            
        - 정책 평가, 정책 개선을 반복하는 dynamic programming
            
    - 실제 환경과 상호작용하지 않고도  
        환경 모델만 가지고 최적 정책을 계산할 수 있다.
        
2. 환경 모델을 경험으로부터 학습하는 경우
    
    - 전이 모델 fθ(s, a) ≈ s', 보상 모델 gθ(s, a) ≈ r 등을  
        신경망이나 다른 함수 근사기로 학습한다.
        
    - 그 다음 이 모델을 사용해  
        가상 roll-out을 생성하고, planning 또는 Model-free 업데이트를 한다.
        
    - 예:
        
        - Dyna-Q:
            
            - 실제 환경에서 얻은 경험으로  
                Q를 업데이트하는 동시에  
                모델 P̂, R̂를 업데이트하고  
                이 모델로 가짜 경험을 생성해  
                Q를 추가로 업데이트한다.
                
        - Model-based RL with MPC:
            
            - 현재 상태에서 여러 행동 시퀀스를  
                learned dynamics로 시뮬레이션한 뒤  
                Return이 높은 시퀀스를 골라  
                첫 번째 행동만 실제로 실행한다.
                

이 접근에서는

- 모델의 정확도
    
- planning 비용(계산량)
    
- 모델 불확실성 처리
    

등이 중요한 이슈가 된다.

---

### 3. Hybrid 접근: Model-free + Model-based

실제 연구와 실무에서는  
둘 중 하나만 쓰기보다  
서로의 장단점을 보완하는 하이브리드 구조가 많이 사용된다.

대표적인 아이디어들:

- Model-free를 메인으로,  
    Model-based는 보조적인 roll-out 생성용으로 사용
    
    - 예: Dyna 계열
        
        - 실제 경험 + 가상 경험을 섞어  
            샘플 효율을 높인다.
            
- Model-based로 rough한 planning을 하고,  
    세부 튜닝은 Model-free로 미세 조정
    
    - 예:
        
        - 고수준 계획(경로, 큰 전략)은 모델 기반
            
        - 저수준 제어(모터 토크, 세밀한 조작)는 Model-free 정책
            
- 불확실한 영역에서만 Model-free 탐색을 강화
    
    - 모델이 자신 있는 영역에서는 모델 기반 계획
        
    - 자신 없는 영역에서는 Model-free 탐색과 학습
        

이런 구조는

- 안전성, 샘플 효율, 최종 성능 사이의 균형을  
    유연하게 조절할 수 있다는 점에서  
    실제 시스템 적용에서 점점 중요해지고 있다.
    

---

### 4. 설계 및 실무 관점 요약

- 환경 시뮬레이터가 매우 정확하고 빠르다면
    
    - 가능한 한 Model-based, planning 위주로 접근하고
        
    - 필요하면 일부 구성 요소에 Model-free를 섞어 쓴다.
        
- 환경이 복잡하고, 정확한 모델을 만들기 어렵다면
    
    - DQN, PPO, SAC 같은 Model-free deep RL을 우선 고려한다.
        
- 데이터 수집이 매우 비싸거나 위험한 경우
    
    - Model-based 또는 하이브리드 접근으로  
        샘플 효율을 높이는 전략이 중요하다.