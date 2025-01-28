import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# Global Constants
WINDOW_SIZE = 500  # Fixed window size (width and height)
GRID_SIZE = 20     # Number of grid cells along one side
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
RENDER_DELAY = 10  # Milliseconds delay for rendering speed

# Actions: Left, Right, Straight
ACTIONS = ['LEFT', 'RIGHT', 'STRAIGHT']
ACTION_NAMES = ACTIONS

class SnakeGame:
    """
    Implements the Snake game mechanics, including snake movement, food placement, and collisions.
    """
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """
        Resets the game to its initial state with the snake at the center and food placed randomly.
        """
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, -1)  # Initial direction: UP
        self.food = self.spawn_food()
        self.done = False
        self.score = 0 # Score initialization
        return self.get_state()

    def spawn_food(self):
        """
        Randomly places food on the grid, ensuring it doesn't overlap with the snake.
        """
        while True:
            food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        """
        Executes one step in the game based on the given action.
        - Updates the snake's direction.
        - Checks for collisions or food consumption.
        - Updates the snake's position.
        """
        if self.done:
            return self.get_state(), 0, True

        # Update direction based on action
        if action == 'LEFT':
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 'RIGHT':
            self.direction = (self.direction[1], -self.direction[0])


        # Calculate new head position
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check for collisions
        if (new_head in self.snake or not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)):
            self.done = True
            return self.get_state(), -10, True

        # Update snake
        self.snake.insert(0, new_head)
        reward = -1  # Small negative reward for each step

        # Check food collision
        if new_head == self.food:
            reward = 10
            self.score += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()

        # Additional Reward
        head = self.snake[0]
        prev_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        if new_distance < prev_distance:
            reward += 1  # Reward for getting closer to food

        return self.get_state(), reward, self.done

    def get_state(self):
        """
        Returns the current state of the game as a simple representation:
        - Snake's direction.
        - Relative position of the food.
        """
        head = self.snake[0]
        food = self.food
        direction = self.direction

        # Relative position of food
        food_dir = (np.sign(food[0] - head[0]), np.sign(food[1] - head[1]))

        # Create state representation
        state = np.array([direction[0], direction[1], food_dir[0], food_dir[1]])
        return state

    def render(self, title="Q-Learning"):
        """
        Renders a single game instance.
        """
        img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        cell_size = WINDOW_SIZE // self.grid_size

        # Draws gridlines for the game
        for i in range(0, WINDOW_SIZE, cell_size):
            cv2.line(img, (i, 0), (i, WINDOW_SIZE), (50, 50, 50), 1)
            cv2.line(img, (0, i), (WINDOW_SIZE, i), (50, 50, 50), 1)

        # Draws the snake
        for x, y in self.snake:
            cv2.rectangle(img, (x * cell_size, y * cell_size),
                        ((x + 1) * cell_size, (y + 1) * cell_size), (0, 255, 0), -1)

        # Draws the food
        fx, fy = self.food
        cv2.rectangle(img, (fx * cell_size, fy * cell_size),
                    ((fx + 1) * cell_size, (fy + 1) * cell_size), (0, 0, 255), -1)

        # Adds the title and border
        cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(img, (0, 0), (WINDOW_SIZE, WINDOW_SIZE), (255, 255, 255), 2)  # Border

        # Display the game
        cv2.imshow("Snake Game", img)
        cv2.waitKey(RENDER_DELAY)



class QLearningAgent:
    """
    Implements a basic Q-Learning agent.
    The agent learns a Q-table to map states and actions to expected rewards.
    """
    def __init__(self, actions, state_size):
        self.actions = actions
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def get_state_key(self, state):
        """
        Converts the state (array) into a hashable key for the Q-table.
        """
        return tuple(state)

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy:
        - Explores (random action) with probability epsilon.
        - Exploits (best known action) otherwise.
        """
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.epsilon or state_key not in self.q_table:
            return random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state_key])]

    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-value for a given state and action using the Q-Learning update rule:
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))
        action_index = self.actions.index(action)
        best_next_action = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action_index] += self.alpha * (reward + self.gamma * best_next_action - self.q_table[state_key][action_index])


if __name__ == "__main__":
    grid_size = 8
    state_size = 4  # Direction + Food direction

    game_q = SnakeGame(grid_size=grid_size)

    agent_q = QLearningAgent(ACTIONS, state_size)

    episodes = 200  # Number of episodes
    metrics = {
        "Q-Learning": [],
        "Food Eaten Q-Learning": [],
        "Steps Q-Learning": []
    }

    for episode in range(episodes):
        state_q = game_q.reset()
        done_q = False
        total_reward_q = 0
        steps_q = 0
        food_eaten_q = 0

        while not done_q:
            action_q = agent_q.choose_action(state_q)
            next_state_q, reward_q, done_q = game_q.step(action_q)
            agent_q.update_q_value(state_q, action_q, reward_q, next_state_q)
            total_reward_q += reward_q
            steps_q += 1
            state_q = next_state_q
            if reward_q == 10:  # Food eaten
                food_eaten_q += 1

            game_q.render()  # Render the game

        # Log metrics for the episode
        metrics["Q-Learning"].append(total_reward_q)
        metrics["Food Eaten Q-Learning"].append(food_eaten_q)
        metrics["Steps Q-Learning"].append(steps_q)

        print(f"Episode {episode + 1}: Q-Learning Reward = {total_reward_q}, Food = {food_eaten_q}, Steps = {steps_q}")

    # Plot metrics
    plt.figure(figsize=(15, 10))

    # Total Reward
    plt.subplot(3, 1, 1)
    plt.plot(metrics["Q-Learning"], label="Q-Learning", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward Comparison")
    plt.legend()

    # Food Eaten
    plt.subplot(3, 1, 2)
    plt.plot(metrics["Food Eaten Q-Learning"], label="Food Eaten Q-Learning", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Food Eaten")
    plt.title("Food Eaten Comparison")
    plt.legend()

    # Steps Taken
    plt.subplot(3, 1, 3)
    plt.plot(metrics["Steps Q-Learning"], label="Steps Q-Learning", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Steps Taken")
    plt.title("Steps Taken Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

    cv2.destroyAllWindows()


