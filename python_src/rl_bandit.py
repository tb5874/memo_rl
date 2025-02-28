import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    # arms = 슬롯머신 수
    def __init__(self, arms):
        self.arms = arms
        self.rates = np.random.rand( self.arms )
            # 0 ~ 1 사이의 랜덤값을 슬롯머신 수 만큼 생성
            # 각 랜덤값은 슬롯머신의 패배 확률
            # 사람 입장에서 (0 ~ 슬롯머신의 패배 확률) 범위안의 값이 나오면 성공

    def play(self, arm):
        rate = self.rates[arm]
        return (1 if rate > np.random.rand() else 0)

    def get_rate(self, arm = -1):
        if (arm == -1):
            np.set_printoptions(suppress=True)
            np.set_printoptions(linewidth=1000)
            for idx in range(0, len(self.rates), 10):
                print(self.rates[idx:idx+10])
            np.set_printoptions()
        else:
            print("bandit    : ", arm)
            print("bandit Q  : ", self.rates[arm])
        print( "" )

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
            # 1행 : 각 슬롯머신의 누적 플레이 횟수
            # 2행 : 각 슬롯머신의 현재 Q(가치=기대값)
        self.epsilon = epsilon
            # 무작위로 행동할 확률(탐색 확률)

    def update(self, arm, reward):
        arm_try = self.info[0,:]
        arm_Q = self.info[1,:]
        arm_try[arm] += 1
        arm_Q[arm] += (reward - arm_Q[arm]) / arm_try[arm]

    def get_action(self):

        case = 0 # [ ε- greedy ]
        case = 1 # [ UCB, Upper Confidence Bound ]

        if (case == 0):
            # [ ε- 탐욕 정책 ]
            arm_Q = self.info[1,:]
            if np.random.rand() < self.epsilon:                
                return np.random.randint(0, len(arm_Q) ) # 무작위 행동
            else:                
                arm_Q = self.info[1,:]
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
                return np.argmax( ucb_values ) # 가장 큰 기대값을 선택
        else:
            print("not yet")

    def get_maxQ(self):
        idx = np.argmax( self.info[1] )
        print( "agent     : ", idx, "(", int(self.info[0][ idx ]),"try )" )
        print( "agent Q   : ", self.info[1][ idx ] )
        print( "" )
        return idx


if __name__ == '__main__':

    #np.random.seed(1234)

    arms = 10
    steps = 10000
    epsilon = 0.1

    bandit = Bandit(arms)
    agent = Agent(arms, epsilon)
    reward_sum = 0 # 누적 보상
    reward_total = [] # 누적 단계별 보상
    rates = [] # 누적 단계별 승률
    act_arm = [] # 누적 단계별 행동

    for step in range(steps):
        action = agent.get_action() # 행동 선택
        reward = bandit.play(action) # 실제로 플레이하고 보상을 받음
        agent.update(action, reward) # 행동과 보상을 통해 각 슬롯머신의 기대치를 갱신 == 학습
        reward_sum += reward # 현재까지 이긴 횟수
        rates.append( reward_sum / (step + 1) ) # 현재까지 승률
        reward_total.append(reward_sum) # 누적 플레이 마다 이긴 횟수 저장
        act_arm.append(action)

    print("total try : ", steps, "\n")

    get_arm = agent.get_maxQ()
    bandit.get_rate(get_arm)
    bandit.top_rate(10)

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