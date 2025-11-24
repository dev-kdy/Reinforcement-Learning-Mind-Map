상위: [[RL 00 - Reinforcement Learning Index]]  
관련: [[RL 08 - Model-free vs Model-based RL]], [[RL 10 - Value Function (V, Q)]], [[RL 12 - Exploration vs Exploitation]], [[RL 14 - Temporal-Difference (TD) Learning]], [[RL 16 - SARSA]], [[RL 17 - Deep Q-Network (DQN)]]

---

## What (정의)

Q-learning은  
가장 대표적인 model-free, off-policy, value-based 강화학습 알고리즘으로,  
최적 행동가치함수 Q*(s, a)를 근사하는 것을 목표로 한다.

핵심 특징을 정리하면 다음과 같다.

- model-free  
    환경의 전이 확률이나 보상 함수를 알 필요 없이,  
    경험 (s, a, r, s') 샘플만으로 학습한다.
    
- off-policy  
    실제로 어떤 행동을 하며 데이터를 모으더라도,  
    업데이트는 항상 "가장 좋은 행동을 했다고 가정한"  
    가상의 greedy 정책을 기준으로 이루어진다.
    
- value-based  
    Q(s, a)라는 Value Function을 먼저 학습하고,  
    그 위에서 greedy 또는 ε-greedy 정책을 유도한다.
    

결국 Q-learning은  
경험으로부터 Q*(s, a)를 점점 근사해 가며,  
그 Q*에 대한 greedy 정책이 최적 정책이 되도록 하는 알고리즘이다.

---

## Why (배경/목적)

Q-learning이 중요한 이유는 다음과 같다.

1. 환경 모델 없이도 최적 정책 학습 가능  
    [[RL 03 - Environment & MDP]]의 전이 모델 P(s' | s, a)를 전혀 몰라도  
    경험 (s, a, r, s')만으로 최적 행동가치함수 Q*를 근사할 수 있다.  
    이것은 [[RL 08 - Model-free vs Model-based RL]]에서 말하는  
    전형적인 model-free 접근의 대표 사례다.
    
2. 이론적으로 잘 정립된 off-policy 알고리즘  
    적절한 조건(탐색, 학습률, Markov성 등) 아래에서는  
    tabular 설정에서 Q-learning이 Q*로 수렴한다는 이론적 결과가 있다.  
    즉, 충분히 오래 학습하면 최적 정책에 가까워질 수 있다는 보장이 있다.
    
3. 구현이 단순하고 직관적  
    업데이트 규칙이 [[RL 14 - Temporal-Difference (TD Learning)]] 형태라  
    코드로 옮기기 쉽고,  
    Q 테이블 하나만 잘 관리하면 되기 때문에  
    입문 및 교육용 예제로 매우 자주 사용된다.
    
4. Deep RL로 자연스럽게 확장 가능  
    상태 공간이 커지면 Q 테이블을 그대로 쓰기 어렵다.  
    이때 Q(s, a)를 신경망으로 근사한 것이  
    [[RL 17 - Deep Q-Network (DQN)]]이다.  
    즉, Q-learning 아이디어 자체가  
    Deep Q-learning, Double DQN, Dueling DQN 등  
    많은 Deep RL 알고리즘의 기반이 된다.
    

요약하면, Q-learning은  
"환경 모델 없이, 간단한 업데이트로 Q*를 학습하는  
가장 기본적인 model-free, off-policy value-based 알고리즘"이다.

---

## How (활용)

### 1. 핵심 아이디어

Q-learning의 아이디어는 간단하다.

- 지금 상태 s에서 행동 a를 했을 때의 Q(s, a)를
    
- 한 스텝 뒤의 보상 r과, 다음 상태 s'에서의 최대 Q값을 이용해
    
- TD 방식으로 조금씩 보정해 나간다.
    

이때 사용되는 패턴은 TD 업데이트의 전형적인 형태다.

- 새로운 값 ← 현재 값 + 학습률 × (목표값 − 현재 값)
    

여기서 목표값은

- 즉시 보상 r
    
- 할인된 미래 최댓값 γ max_{a'} Q(s', a')
    

두 요소로 구성된다.

---

### 2. 알고리즘 흐름 (개념적 단계)

Q-learning의 한 에피소드(또는 한 에이전트-환경 상호작용 루프)는  
대략 다음과 같이 진행된다고 볼 수 있다.

1. 초기화
    
    - Q(s, a)를 임의의 값으로 초기화한다  
        (처음에는 아무것도 모르는 상태).
        
2. 상태 관찰
    
    - 현재 상태 s를 관찰한다.
        
3. 행동 선택 (Exploration vs Exploitation)
    
    - [[RL 12 - Exploration vs Exploitation]]에서 설명한  
        ε-greedy 정책을 사용해 행동 a를 선택한다.
        
        - 확률 ε: 랜덤 행동 (탐색)
            
        - 확률 1−ε: Q(s, a)가 최대인 행동 (활용)
            
4. 한 스텝 진행
    
    - 환경에 행동 a를 적용하고  
        다음 상태 s'와 보상 r을 관측한다.
        
5. Q 업데이트 (TD Learning)
    
    - s, a에 대한 Q(s, a)를  
        TD 형태로 조금 수정한다.
        
    - 이때 목표값에는 s'에서의 max_{a'} Q(s', a')가 들어간다  
        (off-policy 특성).
        
6. 다음 상태로 이동
    
    - s ← s'로 갱신하고,
        
    - 종료 상태가 아니면 3~5 과정을 반복한다.
        

이 과정을 여러 에피소드 동안 반복하면서  
Q(s, a)가 점점 Q*(s, a)에 가까워지도록 학습된다.

---

### 3. Exploration vs Exploitation과의 연계

Q-learning은 기본적으로 greedy 정책을 통해  
Q값을 최대화하는 방향으로 설계되어 있다.  
하지만 탐색이 없으면  
초기에 우연히 크게 나온 행동에만 매달려  
더 좋은 행동을 놓칠 수 있다.

그래서 보통 다음과 같이 한다.

- 행동 선택은 ε-greedy 사용
    
    - [[RL 12 - Exploration vs Exploitation]]에서 설명한 것처럼,  
        일정 확률로 랜덤 행동을 섞어 탐색한다.
        
- ε 스케줄
    
    - 학습 초반에는 ε를 크게 두어  
        다양한 행동을 시도하도록 하고
        
    - 시간이 지날수록 ε를 줄여  
        학습된 Q값을 더 많이 활용하도록 만든다.
        

이렇게 해야  
Q-learning이 Q*에 수렴할 조건(충분한 탐색)을 만족할 수 있다.

---

### 4. SARSA와의 비교 (on-policy vs off-policy)

[[RL 16 - SARSA]]는 Q-learning과 매우 비슷하지만,  
on-policy TD 업데이트를 사용한다.

- SARSA
    
    - 목표값에서 실제로 다음 시점에 선택한 행동 a_{t+1}을 사용한다.
        
    - 즉, 현재 정책이 실제로 걷는 경로를 그대로 따른다.
        
- Q-learning
    
    - 목표값에서 다음 상태의 가능한 행동 중  
        Q가 최대인 행동을 사용한다.
        
    - 실제로는 탐색 때문에 다른 행동을 했더라도,  
        업데이트는 항상 “가장 좋은 행동을 했다고 가정한”  
        이상적인 greedy 정책을 기준으로 한다.
        

이 차이 때문에

- SARSA는 현재의 탐색 정책을 그대로 반영하는 on-policy 학습
    
- Q-learning은 탐색은 따로 하되,  
    업데이트는 이상적인 greedy 정책을 향해 가는 off-policy 학습
    

으로 이해하면 된다.

---

### 5. 고차원 상태에서의 확장: DQN

Q-learning의 기본 형태는  
각 (s, a) 조합에 대해  
Q값을 테이블로 저장하는 tabular 방식이다.

하지만

- 상태가 이미지(픽셀)처럼 고차원인 경우  
    모든 상태를 테이블에 저장하는 것은 불가능하다.
    
- 이때는 [[RL 17 - Deep Q-Network (DQN)]]처럼  
    Q(s, a)를 신경망으로 근사한다.
    

DQN은 다음과 같은 요소를 추가해  
Q-learning을 고차원/복잡한 환경에 적용한다.

- 신경망을 이용한 함수 근사
    
- 경험 재플레이 버퍼로 샘플 decorrelation
    
- 타깃 네트워크로 학습 안정화
    

즉, Q-learning의 기본 아이디어는 그대로 두되  
표 대신 신경망을 쓰고,  
추가적인 테크닉으로 안정성을 확보한 것이 DQN이다.

---

### 6. 설계 및 실무 관점 요약

- Q-learning은  
    model-free, off-policy, value-based RL의 정석적인 출발점이다.
    
- 핵심 구성
    
    - Q(s, a) 테이블 또는 함수 근사기
        
    - ε-greedy 탐색
        
    - TD 형태의 업데이트 규칙
        
- 실무/연구에서의 위치
    
    - 간단한 문제(그리드월드, 작은 MDP)에서는  
        tabular Q-learning만으로도 충분히 유용하다.
        
    - 고차원 문제에서는  
        DQN 계열로 자연스럽게 확장된다.
        
- 이해 포인트
    
    - [[RL 10 - Value Function (V, Q)]], [[RL 14 - Temporal-Difference (TD) Learning]], [[RL 12 - Exploration vs Exploitation]]을  
        Q-learning 관점에서 함께 묶어 보면,  
        이후의 DQN, Double DQN, PPO 등  
        다른 알고리즘 구조를 이해하기가 훨씬 쉬워진다.