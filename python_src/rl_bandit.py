import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    # arms = ���Ըӽ� ��
    def __init__(self, arms):
        self.arms = arms
        self.rates = np.random.rand( self.arms )
            # 0 ~ 1 ������ �������� ���Ըӽ� �� ��ŭ ����
            # �� �������� ���Ըӽ��� �й� Ȯ��
            # ��� ���忡�� (0 ~ ���Ըӽ��� �й� Ȯ��) �������� ���� ������ ����

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
        for idx in range(0, top) :
            print("bandit top(", idx, ") : ", f'{top_idx[idx]:5}', " : ", self.rates[top_idx[idx]])
        np.set_printoptions()

class Agent:
    # ���� ����� ����
    def __init__(self, arms, epsilon):
        self.arms = arms
        self.info = np.zeros((2, self.arms))
            # 1�� : �� ���Ըӽ��� ���� �÷��� Ƚ��
            # 2�� : �� ���Ըӽ��� ���� Q(��ġ=��밪)
        self.epsilon = epsilon
            # �������� �ൿ�� Ȯ��(Ž�� Ȯ��)

    # �ش��ϴ� ���Ըӽ��� ���� �÷��̼��� Q(��ġ=��밪)����
        # �������� ���� ��ġ�� �����ϰ���
        # �ش� ��������
        # ��� ���Ըӽ��� ���� �÷��̼��� �����ϰ� ��ġ����
        # ��ġ, ��밪�� ���ϰ�
        # �� �������� ���� ����ϴ°͵� ����ϵ� ?
    def update(self, arm, reward):
        arm_try = self.info[0,:]
        arm_Q = self.info[1,:]
        arm_try[arm] += 1
        arm_Q[arm] += (reward - arm_Q[arm]) / arm_try[arm]

    # �ൿ ����(��-Ž�� ��å)
    def get_action(self):

        case = 0

        if (case == 0):
            # [ ��-Ž�� ��å ]
            if np.random.rand() < self.epsilon:
                # ������ �ൿ
                return np.random.randint(0, self.arms)
            else:
                # ���� ū ��밪�� �����ϴ� �ൿ
                arm_Q = self.info[1]
                return np.argmax( arm_Q )
        elif(case == 1):
            # [ UCB ]
            arm_try = self.info[0,:]
            arm_Q = self.info[1,:]
            if 0 in arm_try:
                # ��� ���Ըӽ��� �ּ� 1������ ������ �� �ֵ��� Ʈ���� �ش�.
                return np.argmin( arm_try )
            else:
                ucb_values = arm_Q + np.sqrt( (2 * np.log( sum(arm_try) ) / self.counts )
        else:
            pass

    def get_maxQ(self):
        idx = np.argmax( self.info[1] )
        print( "agent     : ", idx, "(", int(self.info[0][ idx ]),"try )" )
        print( "agent Q   : ", self.info[1][ idx ] )
        print( "" )
        return idx


if __name__ == '__main__':

    arms = 100
    steps = 100000
    epsilon = 0.1

    bandit = Bandit(arms)
    agent = Agent(arms, epsilon)
    reward_sum = 0 # ���� ����
    reward_total = [] # ���� �ܰ躰 ����
    rates = [] # ���� �ܰ躰 �·�
    act_arm = [] # ���� �ܰ躰 �ൿ

    for step in range(steps):
        action = agent.get_action() # �ൿ ����
        reward = bandit.play(action) # ������ �÷����ϰ� ������ ����
        agent.update(action, reward) # �ൿ�� ������ ���� �� ���Ըӽ��� ���ġ�� ���� == �н�
        reward_sum += reward # ������� �̱� Ƚ��
        rates.append( reward_sum / (step + 1) ) # ������� �·�
        reward_total.append(reward_sum) # ���� �÷��� ���� �̱� Ƚ�� ����
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
    #for i in range(len(act_arm)):
    #    plt.text(i, act_arm[i], f'{act_arm[i]}', fontsize=5)    
    #plt.yticks(np.arange(0, arms, 1))
    plt.show()    
