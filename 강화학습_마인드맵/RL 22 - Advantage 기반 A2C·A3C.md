상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 21 - Actor-Critic (구조)]], [[RL 11 - Advantage Function]], [[RL 20 - Policy Gradient (기본 PG)]], [[RL 10 - Value Function (V, Q)]]

---

## What (정의)

A2C(Advantage Actor-Critic)와 A3C(Asynchronous Advantage Actor-Critic)는  
[[RL 21 - Actor-Critic (구조)]]에 [[RL 11 - Advantage Function]]을 결합하고,  
여러 환경을 병렬로 돌려 데이터를 모으는 방식의 대표적인 Actor-Critic 계열 알고리즘이다.

조금 더 풀어서 보면 다음과 같다.

- Actor  
    상태를 입력받아 정책 π(a|s)를 출력하는 모듈이다.  
    이 정책으로 행동을 선택한다.
    
- Critic  
    상태 가치 V(s)를 추정하는 모듈이다.  
    이 값을 이용해 Advantage를 계산한다.
    
- Advantage 기반 업데이트  
    A(s, a)를 이용해  
    “이 행동이 그 상태에서 평균보다 얼마나 더 좋았는지”를 정책 업데이트에 반영한다.
    
- 멀티 워커 구조  
    여러 환경 인스턴스를 동시에 실행해  
    다양한 rollout을 모으고,  
    이 데이터를 모아서 Actor와 Critic을 업데이트한다.
    

A3C는 이 구조를 비동기(asynchronous)로,  
A2C는 동기(synchronous)로 구현한 버전이라고 이해하면 된다.

---

## Why (배경/목적)

Advantage 기반 A2C·A3C가 제안된 이유는 크게 두 가지 축으로 볼 수 있다.

1. Advantage로 Policy Gradient 분산 줄이기
    
    - 순수 Policy Gradient([[RL 20 - Policy Gradient (기본 PG)]])는  
        Return 자체를 가중치로 쓰기 때문에  
        학습 신호의 분산이 크고 수렴이 느릴 수 있다.
        
    - Critic이 추정한 V(s)를 기준선으로 빼서  
        A(s, a) = G_t − V(s_t) 같은 Advantage를 사용하면  
        “평균보다 얼마나 더 좋은 행동이었는지”만 남게 되어  
        Policy Gradient 분산이 줄어들고 학습이 더 안정적이 된다.
        
2. 멀티 환경 병렬화로 샘플 효율 향상
    
    - 단일 환경에서 하나의 에이전트만 돌리면  
        데이터 수집 속도가 느리고,  
        특정 궤적에 지나치게 치우친 경험만 보게 될 수 있다.
        
    - 여러 환경 인스턴스를 병렬로 돌리는 구조를 사용하면
        
        - 시간당 더 많은 경험을 수집할 수 있고
            
        - 다양한 초기 조건·궤적에서 나온 데이터를 한 번에 모을 수 있어  
            샘플 효율과 안정성이 함께 올라간다.
            
    - A3C는 이 병렬화를 비동기 방식으로 구현해  
        CPU 여러 코어를 적극적으로 활용하는 것을 목표로 했다.
        

결과적으로 A2C·A3C는

- Actor-Critic 구조
    
- Advantage 기반 Policy Gradient
    
- 멀티 환경 병렬 데이터 수집
    

이라는 세 가지 아이디어를 결합해  
안정성과 효율을 동시에 노리는 구조라고 정리할 수 있다.

---

## How (구조 및 활용)

### 1. 기본 Actor-Critic 루프에 Advantage 결합

A2C·A3C의 한 워커에서 일어나는 기본 흐름은 다음과 같이 이해하면 된다.

1. 상태 s_t 관찰
    
2. Actor의 정책 π(a|s_t)에서 행동 a_t 샘플링
    
3. 환경에서 a_t 실행 → 보상 r_{t+1}, 다음 상태 s_{t+1} 관찰
    
4. 일정 길이 T만큼 rollout을 쌓는다
    
    - (s_t, a_t, r_{t+1}), (s_{t+1}, a_{t+1}, r_{t+2}), …
        
5. Critic의 V(s)를 이용해
    
    - Return 또는 n-step Return을 계산하고
        
    - Advantage A_t를 추정한다.
        
6. Advantage를 이용해
    
    - Actor는 Policy loss를 줄이는 방향으로 업데이트
        
    - Critic은 Value loss를 줄이는 방향으로 업데이트
        

Actor-Critic([[RL 21 - Actor-Critic (구조)]])와 구조는 동일하지만,  
여기서 Advantage를 명시적으로 사용한다는 점이 핵심이다.

---

### 2. A2C vs A3C

둘은 아이디어는 거의 같고,  
“여러 워커를 어떻게 동기화하느냐”에서 차이가 난다.

1. A3C (Asynchronous A2C)
    
    - 여러 워커가 각자 에피소드를 돌리며  
        자신의 로컬 네트워크(또는 로컬 파라미터 복사본)를 사용해 rollout을 수집한다.
        
    - 각 워커는 일정 스텝 또는 일정 에피소드마다  
        전역 파라미터에 대해 비동기적으로 업데이트를 적용한다.
        
    - 장점
        
        - CPU 멀티코어를 적극 활용
            
        - 서로 다른 워커들이 서로 다른 시점에서 업데이트해  
            경험 다양성이 높아지는 효과
            
    - 단점
        
        - 구현이 상대적으로 복잡
            
        - 비동기 업데이트로 인해 재현성과 디버깅이 어려울 수 있다.
            
2. A2C (Advantage Actor-Critic, 동기식 버전)
    
    - 여러 워커가 동시에 각자 rollout을 모은 뒤,  
        한 번에 모아서 업데이트를 수행한다.
        
    - 예: N개의 환경을 병렬로 돌려  
        T 스텝의 trajectory를 쌓고,  
        그 모든 데이터를 모아 한 번의 대규모 배치 업데이트를 수행.
        
    - 장점
        
        - 구현이 A3C보다 간단
            
        - 동기식이기 때문에 재현성, 디버깅이 더 수월
            
        - 실제로는 A3C와 성능이 비슷하거나 더 좋은 경우도 많다.
            

요약하면

- 아이디어: Advantage 기반 Actor-Critic
    
- 비동기 구현: A3C
    
- 동기, 배치형 구현: A2C
    

라고 정리할 수 있다.

---

### 3. Advantage 추정 방식

A2C·A3C에서 Advantage A_t는 보통 다음과 같이 추정한다.

1. n-step Return 기반
    
    - 시점 t에서 T까지의 rollout을 쌓았다고 할 때,  
        n-step Return 형태로  
        R_t ≈ r_{t+1} + γ r_{t+2} + … + γ^{k−1} r_{t+k} + γ^k V(s_{t+k})  
        같은 값을 계산한다.
        
    - Advantage는  
        A_t = R_t − V(s_t)  
        처럼 만든다.
        
2. 단순한 형태
    
    - 한 스텝 TD Advantage  
        A_t ≈ r_{t+1} + γ V(s_{t+1}) − V(s_t)
        
    - 또는 몬테카를로 Return 기반  
        A_t ≈ G_t − V(s_t)
        

실제로는 여러 스텝 정보를 섞어  
더 부드럽고 분산이 낮은 Advantage를 만들기도 한다  
(예: GAE와 같은 방식).

중요한 포인트는

- Critic이 제공하는 V(s)를 기준선으로 삼고,
    
- 그 기준선 대비 Return이 얼마나 더 좋았는지를  
    A_t에 담아서
    
- Actor의 Policy Gradient에 곱해 준다는 것이다.
    

---

### 4. 손실 구성 (개념 정리)

A2C·A3C에서 한 워커가 모은 rollout으로  
네트워크를 업데이트할 때,  
손실은 보통 다음 세 가지 항목으로 구성된다.

1. Policy loss (Actor 쪽)
    
    - Advantage가 큰 행동의 로그 확률을 키우는 방향,  
        Advantage가 작은 행동은 줄이는 방향으로 설계된다.
        
    - 개념적으로는  
        Loss_policy ≈ − E[ A_t · log π_θ(a_t | s_t) ]
        
2. Value loss (Critic 쪽)
    
    - V(s_t)가 Return 또는 TD 타깃에 근접하도록 만드는 제곱 오차.
        
    - 개념적으로는  
        Loss_value ≈ E[ (R_t − V_w(s_t))² ]
        
3. Entropy bonus (선택적)
    
    - 정책의 엔트로피를 높이는 항을 추가해  
        너무 일찍 한 행동에만 몰리지 않도록 탐색을 유지한다.
        
    - 개념적으로는  
        Loss_entropy ≈ − β · H(π_θ(·|s_t))
        
    - 전체 손실에서는  
        −Loss_entropy 형태로 들어가  
        엔트로피가 커지도록 유도한다.
        

최종 손실은  
Policy loss + c1 · Value loss − c2 · Entropy term  
와 같이 가중합으로 구성되는 경우가 많다.

---

### 5. 설계 및 실무 관점 요약

- 구조
    
    - Actor-Critic 구조 위에  
        Advantage 기반 Policy Gradient를 얹고  
        멀티 환경 병렬화를 적용한 패밀리다.
        
- A2C vs A3C
    
    - A3C: 비동기, 각 워커가 독립적으로 업데이트
        
    - A2C: 동기, 여러 워커를 모아 배치 업데이트
        
    - 요즘 구현에서는 A2C 스타일이 구현 단순성과 재현성 측면에서 더 많이 쓰이는 편이다.
        
- 장점
    
    - Advantage로 Policy Gradient 분산 감소
        
    - TD 기반 Critic으로 샘플 효율 증가
        
    - 멀티 환경 병렬화로 데이터 수집 속도 향상
        
- 단점 및 주의점
    
    - 하이퍼파라미터(rollout 길이 T, 학습률, γ, 손실 가중치 등) 튜닝 필요
        
    - Actor와 Critic이 서로 영향을 주기 때문에  
        둘 중 하나가 너무 강하거나 약해지지 않도록 균형이 중요하다.
        
- 큰 그림에서의 위치
    
    - [[RL 21 - Actor-Critic (구조)]]의 구체적인 구현 예시이자,
        
    - [[RL 23 - Proximal Policy Optimization (PPO)]] 같은  
        더 발전된 정책 기반 알고리즘으로 넘어가기 전에  
        이해해 두면 좋은 대표적인 Advantage 기반 Actor-Critic 알고리즘이다.