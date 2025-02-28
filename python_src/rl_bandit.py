import time
import numpy as np
import matplotlib.pyplot as plt


class Bandit:    
    def __init__(self, arms):
        np.random.seed(1234) # 생성된 Bandit 머신의 확률을 매 실험마다 고정시키기 위한 트릭             
        self.arms = arms # arms = 슬롯머신 수
        self.rates = np.random.rand( self.arms )
            # 0 ~ 1 사이의 랜덤값을 슬롯머신 수 만큼 생성
            # 각 랜덤값은 슬롯머신의 패배 확률
            # 사람 입장에서 (0 ~ 슬롯머신의 패배 확률) 범위안의 값이 나오면 성공

    def play(self, arm):
        rate = self.rates[arm]
        return (1 if rate > np.random.rand() else 0)

    def get_rate(self, arm = -1, print_flag = True):
        if (arm == -1):
            np.set_printoptions(suppress=True)
            np.set_printoptions(linewidth=1000)
            for idx in range(0, len(self.rates), 10):
                print(self.rates[idx:idx+10])
            np.set_printoptions()
            print( "" )
        else:
            if ( print_flag == True ):
                print("bandit    : ", arm, "\n", end ="" )
                print("bandit Q  : ", self.rates[arm], "\n", end ="" )
        return self.rates

    def top_rate(self, top):
        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=1000)
        top_idx = np.argsort(self.rates)[-10:][::-1]
        top_range = (top if len(self.rates) > top else len(self.rates))
        for idx in range(0, top_range) :
            print("bandit top(", idx, ") : ", f'{top_idx[idx]:5}', " : ", self.rates[top_idx[idx]])
        np.set_printoptions()

class Agent:
    # 정보 저장용 변수
    def __init__(self, arms, epsilon):
        self.arms = arms
        self.info = np.zeros((2, self.arms))
            # index 0 : 각 슬롯머신의 누적 플레이 횟수
            # index 1 : 각 슬롯머신의 현재 Q(가치, 기대값)
        self.epsilon = epsilon
            # 무작위로 행동할 확률(탐색 확률)

    def update(self, arm, reward):
        arm_try = self.info[0,:]
        arm_Q = self.info[1,:]
        arm_try[arm] += 1
        arm_Q[arm] += (reward - arm_Q[arm]) / arm_try[arm]

    def get_action(self):

        np.random.seed( int(time.perf_counter() * 1e9) % (2**32) ) # 행동 선택 시, 랜덤 요소를 정말 랜덤하게 가져가기 위함

        case = 0 # [ ε- greedy ]
        case = 1 # [ UCB, Upper Confidence Bound ]
        case = 2 # [ UCB + ε- greedy ]

        if (case == 0):
            # [ ε- greedy ]
            arm_Q = self.info[1,:]
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, len(arm_Q) ) # 무작위 행동
            else:                
                return np.argmax( arm_Q ) # 가장 큰 기대값을 선택
        elif(case == 1):
            # [ UCB ]
            arm_try = self.info[0,:]
            arm_Q = self.info[1,:]
            total_try = sum(arm_try)
            if 0 in arm_try:
                return np.argmax( arm_try == 0 ) # 모든 슬롯머신이 최소 1번씩은 수행할 수 있도록 트릭을 준다.
            else:
                ucb_values = arm_Q + np.sqrt( (2 * np.log( total_try )) / arm_try  ) # 모든 슬롯머신의 UCB 값을 구한다.
                return np.argmax( ucb_values ) # 가장 큰 UCB값을 선택
        elif(case == 2):
            # [ UCB + ε- greedy ]
            arm_try = self.info[0,:]
            arm_Q = self.info[1,:]
            total_try = sum(arm_try)
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, len(arm_Q) ) # 무작위 행동
            else:                
                if 0 in arm_try:
                    return np.argmax( arm_try == 0 ) # 모든 슬롯머신이 최소 1번씩은 수행할 수 있도록 트릭을 준다.
                else:
                    ucb_values = arm_Q + np.sqrt( (2 * np.log( total_try )) / arm_try  ) # 모든 슬롯머신의 UCB 값을 구한다.
                    return np.argmax( ucb_values ) # 가장 큰 UCB값을 선택
        else:
            print("not yet")

    def get_maxQ(self, prnit_flag = True):
        idx = np.argmax( self.info[1] )
        if( prnit_flag == True):            
            print( "agent     : ", idx, "(", int(self.info[0][ idx ]),"try )\n", end ="" )
            print( "agent Q   : ", self.info[1][ idx ], "\n", end ="" )
        return idx

    def get_Q(self, prnit_flag = True):
        target = self.info[1]
        if ( prnit_flag == True ):
            np.set_printoptions(suppress=True)
            np.set_printoptions(linewidth=1000)            
            for idx in range(0, len(target), 10):
                print(target[idx:idx+10])
            np.set_printoptions()
        return target

def top_rate(title, input, top):
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=1000)
    top_idx = np.argsort(input)[-10:][::-1]
    top_range = (top if len(input) > top else len(input))
    for idx in range(0, top_range) :
        print(title, "top(", idx, ") : ", f'{top_idx[idx]:5}', " : ", input[top_idx[idx]], "\n", end ="" )
    np.set_printoptions()

# Unit Test 의 그래프는 경향성을 보기 위해서 필요하다.
def Test01():

    arms = 100
    steps = 10000

    bandit = Bandit(arms)
    agent = Agent(arms, 0.1)
    reward_sum = 0 # 누적 보상
    reward_total = [] # 누적 단계별 보상
    rates = [] # 누적 단계별 승률
    act_arm = [] # 누적 단계별 행동

    for step in range(steps):
        action = agent.get_action() # 행동 선택
        reward = bandit.play(action) # 실제로 플레이하고 보상을 받음
        agent.update(action, reward) # 행동과 보상을 통해 각 슬롯머신의 기대치를 갱신 == 학습
        reward_sum += reward # 현재까지 이긴 횟수
        rates.append( reward_sum / (step + 1) ) # 누적 플레이 마다 승률(여기서는 기대값)
        reward_total.append(reward_sum)         # 누적 플레이 마다 이긴 횟수 저장
        act_arm.append(action)                  # 누적 플레이 마다 선택한 행동

    print("total try : ", steps, "\n", end = "")
    arm_maxQ = agent.get_maxQ()
    top_rate("bandit", bandit.get_rate(arm_maxQ), 10)

    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(reward_total)
    plt.show()

    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()

    plt.ylabel('Action')
    plt.xlabel('Steps')
    plt.scatter(range(len(act_arm)), act_arm, s=1)
    plt.show()

    return

# Batch Test 의 그래프는 의미가 없다 ?
def Test02(batch):    

    arms = 100
    steps = 10000

    bandit = Bandit(arms)

    arm_batchQ = []

    for batch_setp in range(batch):
        agent = Agent(arms, 0.1)
        for step in range(steps):
            action = agent.get_action() # 행동 선택
            reward = bandit.play(action) # 실제로 플레이하고 보상을 받음
            agent.update(action, reward) # 행동과 보상을 통해 각 슬롯머신의 기대치를 갱신 == 학습
        arm_batchQ.append( agent.get_Q(False) )
        print(f"\rbatch : {batch_setp}", end="")    
    print("\n", end ="" )

    arm_meanQ = np.mean(arm_batchQ, axis=0)
    top_rate("agent", arm_meanQ,10)

    return

if __name__ == '__main__':

    Test01()
    Test02(100)
