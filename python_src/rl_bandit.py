import time
import numpy as np
import matplotlib.pyplot as plt

def util_top(input, top):
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=1000)
    top_idx = np.argsort(input)[-10:][::-1]
    top_range = (top if len(input) > top else len(input))
    for idx in range(0, top_range) :
        print("top(", idx, ") : ", f'{top_idx[idx]:5}', " : ", input[top_idx[idx]], "\n", end ="" )
    np.set_printoptions()
    return

class Bandit :
    def __init__(self, arms):
        np.random.seed(1234) # 생성하는 Bandit 머신의 확률을 매 실험마다 고정시키기 위한 트릭
        self.arms = arms # 슬롯머신 수
        self.rates = np.random.rand( self.arms )
            # 0 ~ 1 사이의 랜덤값을 슬롯머신 수 만큼 생성
            # 각 랜덤값은 슬롯머신의 패배 확률
            # 사람 입장에서 (0 ~ 슬롯머신의 패배 확률) 범위안의 값이 나오면 성공

    def play(self, arm):
        rate = self.rates[arm]
        #self.rates += 0.1 * np.random.randn(self.arms)
        return (1 if rate > np.random.rand() else 0)

    def rate_all(self):
        return self.rates

    def rate_arm(self, arm, flag_show = True):
        if ( flag_show == True ):
            print("bandit -", arm, ":", self.rates[arm], "\n", end ="" )
        return self.rates[arm]

class Agent :
    def __init__(self, arms, epsilon):
        self.arms = arms # 평가 할 슬롯머신 수
        self.info = np.zeros((2, self.arms))
            # index 0 행 : 각 슬롯머신의 누적 플레이 횟수
            # index 1 행 : 각 슬롯머신의 가치, 기대값 ( Q )
        self.epsilon = epsilon # 무작위로 행동할 확률 ( ε )

    def update(self, arm, reward):
        arms_TRY = self.info[0,:]
        arms_Q = self.info[1,:]
        arms_TRY[arm] += 1
        arms_Q[arm] += (reward - arms_Q[arm]) / arms_TRY[arm]
        return

    def get_ACT(self):

        case = 0 # [ ε- greedy ]
        case = 1 # [ UCB, Upper Confidence Bound ]

        arms_TRY = self.info[0,:]
        arms_Q = self.info[1,:]

        if (case == 0):
            # [ ε- greedy ]
            np.random.seed( int(time.perf_counter() * 1e9) % (2**32) ) # 매 실험마다 다른 결과를 만들기 위함
            if (self.epsilon > np.random.rand()) : return np.random.randint(0, len(arms_Q)) # 무작위 행동
            else : return np.argmax(arms_Q) # 가장 큰 기대값을 선택
        elif (case == 1):
            # [ UCB ]
            if 0 in arms_TRY:
                return np.argmax(arms_TRY == 0) # 모든 슬롯머신이 최소 1번 수행하도록 트릭을 준다.
            else:
                try_total = sum(arms_TRY)
                arms_ucb = arms_Q + np.sqrt( ( 2 * np.log(try_total) ) / arms_TRY  ) # 모든 슬롯머신의 UCB 값을 구한다.
                return np.argmax( arms_ucb ) # 가장 큰 UCB 값 선택
        else:
            print("not yet")

        return

    def get_Q(self):
        return self.info[1]

    def get_maxQ(self, flag_show = True):
        maxQ = np.argmax( self.info[1] )
        if( flag_show == True ):
            print( "agent -", maxQ, "(", int(self.info[0][ maxQ ]),"try ) :",self.info[1][ maxQ ] ,"\n", end ="" )
        return maxQ

# Test 01
# 1회 반복의 그래프는 경향성을 보기 위해 필요하다.
def Test01():

    arms = 100
    steps = 10000

    bandit = Bandit(arms)
    agent = Agent(arms, 0.1)    

    ACT_step = [] # 단계별 선택 행동
    R_total = 0 # 누적 보상
    R_step = [] # 모든 슬롯머신을 사용하며 누적된 보상
    Q_step = [] # 모든 슬롯머신을 사용하며 얻은 기대값

    for step in range(steps):
        ACT = agent.get_ACT() # 정책( 기대값, 반복 횟수, 누적 행동 선택 횟수, 무작위 등... )을 기반으로 행동 선택
        reward = bandit.play(ACT) # 선택한 행동에 대한 보상
        agent.update(ACT, reward) # 보상으로 선택한 행동의 기대값을 갱신 == 학습

        ACT_step.append(ACT)
        R_total += reward
        R_step.append(R_total)
        Q_step.append(R_total / (step + 1))

    print("total try : ", steps, "\n", end = "")
    agent_maxQ = agent.get_maxQ()
    bandit_Q = bandit.rate_arm(agent_maxQ)

    print("[ bandit ]\n", end = "")
    bandit_allQ = bandit.rate_all()
    util_top(bandit_allQ, 10)

    print("[ agent ]\n", end = "")
    agent_allQ = agent.get_Q()
    util_top(agent_allQ, 10)

    plt.ylabel('Reward')
    plt.xlabel('Steps')
    plt.plot(R_step)
    plt.show()

    plt.ylabel('Q')
    plt.xlabel('Steps')
    plt.plot(Q_step)
    plt.show()

    plt.ylabel('Action')
    plt.xlabel('Steps')
    plt.scatter(range( len(ACT_step) ), ACT_step, s=1)
    plt.show()

    return

# Test 02
# n 회 반복의 그래프는 무슨 의미 ?
# 편차 ?
def Test02(batch):    

    arms = 100
    steps = 10000

    bandit = Bandit(arms)
    agent_Q = []

    # need to Parallelization : -->
    for batch_step in range(batch):
        agent = Agent(arms, 0.1)
        for step in range(steps):
            ACT = agent.get_ACT()
            reward = bandit.play(ACT)
            agent.update(ACT, reward)
        agent_Q.append( agent.get_Q() )
        print(f"\rbatch : {batch_step+1}", end="")
    print("\n", end ="" )
    # need to Parallelization : <--    
    
    print("[ agent ( step :", steps,", batch :", batch,") ]\n", end = "")
    agent_meanQ = np.mean(agent_Q, axis=0)
    util_top(agent_meanQ, 10)

    return

if __name__ == '__main__':

    Test01()
    Test02(100)
