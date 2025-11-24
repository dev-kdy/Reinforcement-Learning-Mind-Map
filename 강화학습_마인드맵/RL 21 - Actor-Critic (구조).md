상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 20 - Policy Gradient (기본 PG)]], [[RL 10 - Value Function (V, Q)]], [[RL 11 - Advantage Function]], [[RL 13 - Monte Carlo Learning]], [[RL 14 - Temporal-Difference (TD) Learning]], [[RL 22 - Advantage 기반 A2C·A3C]], [[RL 23 - Proximal Policy Optimization (PPO)]]

---

## What (정의)

Actor-Critic은  
Actor(정책)와 Critic(가치 함수)를 동시에 학습하는 구조의 알고리즘 패밀리다.

- Actor
    
    - 정책 π_θ(a|s)를 담당하는 모듈이다.
        
    - 상태 s를 입력으로 받아, 어떤 행동 a를 할지에 대한 확률 분포 또는 결정적 행동을 출력한다.
        
    - 목표는 기대 Return을 높이는 방향으로 정책 파라미터 θ를 업데이트하는 것이다.
        
- Critic
    
    - 가치 함수 V_w(s) 또는 Q_w(s, a)를 추정하는 모듈이다.
        
    - Actor가 만든 행동이 얼마나 좋았는지 평가하는 역할을 한다.
        
    - [[RL 14 - Temporal-Difference (TD) Learning]] 기반의 업데이트로 학습되는 경우가 많다.
        

Actor-Critic 구조에서는  
Critic이 제공하는 가치/Advantage 정보를 활용해  
Actor의 정책 Gradient를 계산한다.  
즉, 정책 기반 방법([[RL 20 - Policy Gradient (기본 PG)]])과  
가치 기반 방법([[RL 10 - Value Function (V, Q)]])을 결합한 형태라고 볼 수 있다.

---

## Why (배경/목적)

순수 Policy Gradient(REINFORCE 등)는  
직접 Return을 사용하기 때문에 개념은 단순하지만, 다음 한계가 있다.

- Return의 분산이 크다
    
    - 에피소드 전체의 G_t를 그대로 쓰면  
        같은 상태·행동이라도 결과가 크게 출렁일 수 있다.
        
    - 이로 인해 학습이 느리고 불안정해질 수 있다.
        
- 샘플 효율이 낮다
    
    - 에피소드 단위 Monte Carlo 방식이라  
        한 번 실행해 본 trajectory의 정보를  
        충분히 재사용하기 어려운 경우가 많다.
        

Actor-Critic은 Critic을 도입해 이 문제를 완화한다.

- Critic이 V(s) 또는 Q(s, a)를 추정해  
    더 안정적인 가치 정보를 제공한다.
    
- Advantage `A(s, a)`([[RL 11 - Advantage Function]])를 사용해  
    “평균보다 얼마나 더 좋았는지”만 강조함으로써  
    Policy Gradient의 분산을 낮춘다.
    
- TD 업데이트([[RL 14 - Temporal-Difference (TD) Learning]])를 통해  
    한 스텝마다 학습이 가능해져  
    샘플 효율이 올라간다.
    

결과적으로 Actor-Critic 구조는

- Policy Gradient의 장점  
    (연속 행동, 확률적 정책, 직접적인 정책 최적화)와
    
- Value-based의 장점  
    (부트스트랩, 샘플 효율, 안정된 가치 추정)
    

을 함께 얻기 위한 대표적인 설계라고 할 수 있다.

---

## How (구조 및 활용)

### 1. 기본 구조

Actor-Critic 구조는 보통 다음 두 가지 구현 패턴 중 하나를 사용한다.

1. 공유 본체 + 두 개의 head
    
    - 하나의 신경망 본체가 상태 s를 인코딩한다.
        
    - 그 위에
        
        - Policy head: π_θ(a|s)를 출력
            
        - Value head: V_w(s)를 출력  
            두 개의 출력 헤드를 둔다.
            
    - 인코딩 부분을 공유함으로써  
        표현 학습을 재사용하면서  
        정책과 가치 추정을 동시에 학습한다.
        
2. 완전히 분리된 두 네트워크
    
    - Actor 네트워크: 정책 π_θ(a|s) 전용
        
    - Critic 네트워크: 가치 함수 V_w(s) 또는 Q_w(s, a) 전용
        
    - 구조가 명확하고 튜닝이 직관적이지만,  
        파라미터 수가 늘어날 수 있다.
        

어떤 방식을 택하든 핵심은

- Actor는 정책 업데이트
    
- Critic은 가치/Advantage 추정
    

이라는 역할 분담이다.

---

### 2. 학습 루프 (개념 흐름)

Actor-Critic 계열 알고리즘의 전형적인 흐름은 다음과 같이 정리할 수 있다.

1. 상태 관찰
    
    - 환경에서 현재 상태 s_t를 관찰한다.
        
2. Actor로 행동 선택
    
    - Actor의 정책 π_θ(a|s_t)에 따라  
        행동 a_t를 샘플링하거나 결정한다.
        
    - 이때 [[RL 12 - Exploration vs Exploitation]] 측면에서  
        확률적 정책 자체가 탐색을 포함한다.
        
3. 환경 상호작용
    
    - 행동 a_t를 환경에 적용하고
        
    - 보상 r_{t+1}과 다음 상태 s_{t+1}을 관찰한다.
        
4. Critic 업데이트
    
    - Critic은 V_w(s_t) 또는 Q_w(s_t, a_t)를  
        TD 방식으로 업데이트한다.
        
    - 예시 개념
        
        - 상태 가치 기반
            
            - 목표값: r_{t+1} + γ V_w(s_{t+1})
                
            - TD 오차: δ_t = r_{t+1} + γ V_w(s_{t+1}) − V_w(s_t)
                
            - Value loss: TD 오차의 제곱을 줄이도록 w를 학습
                
        - 행동 가치 기반도 유사한 구조로 볼 수 있다.
            
5. Advantage 계산
    
    - Critic의 추정치를 사용해 Advantage를 만든다.
        
    - 예:
        
        - A_t ≈ r_{t+1} + γ V_w(s_{t+1}) − V_w(s_t)  
            또는
            
        - A_t ≈ G_t − V_w(s_t)
            
    - 이는 해당 상태에서 해당 행동이  
        평균적인 기대값보다 얼마나 더 나았는지를 나타낸다.
        
6. Actor 업데이트
    
    - Actor는 Policy Gradient 방향으로  
        정책 파라미터 θ를 갱신한다.
        
    - 개념적으로는
        
        - Advantage가 큰 행동은  
            해당 행동의 로그 확률을 더 키우도록 θ를 업데이트
            
        - Advantage가 작은 또는 음수인 행동은  
            그 행동의 로그 확률이 줄어들도록 업데이트
            
    - 정책 loss는 보통  
        Advantage에 비례한 형태로 정의한다.
        
7. 반복
    
    - 위 과정을 에피소드 또는 여러 스텝에 걸쳐 반복하며  
        Actor와 Critic을 함께 발전시킨다.
        

---

### 3. Advantage와의 관계

Actor-Critic의 핵심 중 하나는  
정책 업데이트에서 Advantage([[RL 11 - Advantage Function]])를 사용하는 것이다.

- 단순히 Return이나 Q만 쓰면  
    학습 신호의 분산이 커진다.
    
- Critic이 제공하는 V(s)를 기준선으로 빼서  
    A(s, a) = Q(s, a) − V(s) 또는  
    A_t ≈ G_t − V(s_t)처럼  
    상대적인 우수성만 반영하면  
    정책 Gradient의 분산을 크게 줄일 수 있다.
    

정책 업데이트는 개념적으로

- Advantage가 큰 행동
    
    - 이런 행동을 더 자주 선택하도록  
        정책 파라미터를 조정
        
- Advantage가 작은 행동
    
    - 이런 행동의 선택 확률을 줄이거나  
        거의 바꾸지 않도록 조정
        

하는 방향으로 이뤄진다.

이 구조가 [[RL 22 - Advantage 기반 A2C·A3C]]와  
[[RL 23 - Proximal Policy Optimization (PPO)]] 등  
현대 Actor-Critic 계열 알고리즘의 공통 기반이다.

---

### 4. 대표 변형과 확장

Actor-Critic 구조는 다양한 알고리즘으로 확장된다.

- A2C / A3C([[RL 22 - Advantage 기반 A2C·A3C]])
    
    - 여러 워커가 병렬로 trajectory를 수집하고
        
    - Advantage 기반의 Actor-Critic 업데이트를 수행하는 구조다.
        
- PPO([[RL 23 - Proximal Policy Optimization (PPO)])
    
    - 정책 업데이트 폭을 제한하는 클리핑 기법으로  
        안정성과 성능을 동시에 노리는 Actor-Critic 계열 알고리즘이다.
        
    - 정책 Gradient × Advantage 구조는 유지하되,  
        정책 변화가 너무 크지 않도록 목적 함수를 조정한다.
        
- 연속 제어 알고리즘들
    
    - DDPG, TD3, SAC 등도  
        넓은 의미에서 Actor-Critic 구조를 취한다.
        
    - Actor는 연속 행동을 출력하고
        
    - Critic은 Q(s, a)를 평가하는 역할을 한다.
        

이처럼 Actor-Critic은  
정책 기반과 가치 기반을 연결하는 공통 골격으로서  
다양한 RL 알고리즘의 뼈대가 된다.

---

### 5. 설계 및 실무 관점 요약

- Actor-Critic의 핵심
    
    - Actor: 정책 최적화
        
    - Critic: 가치/Advantage 추정
        
    - 둘을 함께 학습해서  
        정책 Gradient의 분산을 줄이고  
        샘플 효율을 높이는 구조다.
        
- 구현 시 고려 포인트
    
    - 하나의 네트워크에 두 head를 둘지,  
        두 개의 네트워크로 분리할지 선택
        
    - Value loss와 Policy loss의 비율(가중치)을  
        어떻게 둘지 결정
        
    - 엔트로피 보너스를 추가해  
        탐색을 유지할지 여부 판단
        
    - TD 방식, n-step Return, GAE 등  
        Critic 업데이트 방식을 어떻게 설계할지 결정
        
- 큰 그림에서의 위치
    
    - [[RL 20 - Policy Gradient (기본 PG)]]에서 출발해
        
    - [[RL 10 - Value Function (V, Q)]], [[RL 11 - Advantage Function]]을 결합한 형태가 Actor-Critic이고
        
    - 여기에 병렬화, 클리핑, 다양한 안정화 기법을 더한 것이  
        A2C/A3C, PPO, 기타 최신 Actor-Critic 계열 알고리즘이라고 정리할 수 있다.