import numpy as np
from tqdm import tqdm

# from tqdm.notebook import tqdm


class Q_Learning:
    def __init__(
        self,
        env,
        alpha,
        gamma,
        epsilon,
        numberEpisodes,
        timeSteps,
        numSpaces,
        numberOfBins,
        lowerBounds,
        upperBounds,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        self.timeSteps = timeSteps
        self.numSpaces = numSpaces
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds

        self.sumRewardsEpisode = []

        # this matrix is the action value function matrix
        self.Qmatrix = np.random.uniform(
            low=0, high=1, size=(numberOfBins[0], numberOfBins[1], self.actionNumber)
        )

    def returnIndexState(self, state):
        temp = state[0]
        humid = state[1]

        # tempBins = [np.linspace(0, 50, self.numberOfBins[0]) for _ in range(self.numSpaces)]
        # humidBins = [np.linspace(0, 100, self.numberOfBins[1]) for _ in range(self.numSpaces)]
        # indexTemp = [np.maximum(np.digitize(temp[i], tempBins[i])-1, 0) for i in range(self.numSpaces)]
        # indexHumid = [np.maximum(np.digitize(humid[i], humidBins[i])-1, 0) for i in range(self.numSpaces)]

        tempBins = np.linspace(
            self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0]
        )
        humidBins = np.linspace(
            self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1]
        )

        indexTemp = np.maximum(np.digitize(state[0], tempBins) - 1, 0)
        indexHumid = np.maximum(np.digitize(state[1], humidBins) - 1, 0)

        return tuple([indexTemp, indexHumid])
        # return indexTemp

    def selectAction(self, state, index):
        if index < 500:
            return np.random.choice(self.actionNumber)

        randomNumber = np.random.random()

        # after 7000 episodes, we slowly start to decrease the epsilon parameter
        if index > 7000:
            self.epsilon = 0.999 * self.epsilon

        if randomNumber < self.epsilon:
            return np.random.choice(self.actionNumber)
        else:
            return self.greedy_policy(state)

    def simulateEpisodes(self):
        for indexEpisode in tqdm(range(self.numberEpisodes)):
            rewardsEpisode = []
            stateS = self.env.reset()

            # print("Simulating episode {}".format(indexEpisode))

            for step in range(self.timeSteps):
                stateSIndex = self.returnIndexState(stateS)
                actionA = self.selectAction(stateS, indexEpisode)

                stateSnext, reward, terminated, info = self.env.step(actionA)
                rewardsEpisode.append(reward)

                stateSnextIndex = self.returnIndexState(stateSnext)

                Qmaxnext = np.max(self.Qmatrix[stateSnextIndex])

                if terminated:
                    break

                error = (
                    reward + self.gamma * Qmaxnext - self.Qmatrix[stateSIndex][actionA]
                )
                self.Qmatrix[stateSIndex][actionA] = (
                    self.Qmatrix[stateSIndex][actionA] + self.alpha * error
                )
                # else:
                #     error=reward-self.Qmatrix[stateSIndex+(actionA,)]
                #     self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error

                # set the current state to the next state
                stateS = stateSnext

            # print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

    def greedy_policy(self, state):
        # print(state)
        # print(self.returnIndexState(state))
        # print(self.Qmatrix[self.returnIndexState(state)])
        # print(np.max(self.Qmatrix[self.returnIndexState(state)]))
        action = np.random.choice(
            np.where(
                self.Qmatrix[self.returnIndexState(state)][:]
                == np.max(self.Qmatrix[self.returnIndexState(state)][:])
            )[0]
        )
        return action
