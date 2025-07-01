import random
import numpy as np

GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
DIRECTION_VECTORS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
}

class SnakeEnv:
    def __init__(self):
        self.reset()
        self.bonus_active = False
        self.bonus_food = None

    def reset(self):
        self.snake = [(5, 5)]
        self.direction = 'RIGHT'
        self.food = self.spawn_food()
        self.score = 0
        self.done = False
        self.bonus_active = False
        self.bonus_food = None
        return self.get_state()

    def spawn_food(self):
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in self.snake:
                return pos

    def step(self, action_index):
        if self.done:
            return self.get_state(), 0, True

        action = ACTIONS[action_index]
        if self.is_opposite(self.direction, action):
            action = self.direction  # Ignore reverse direction
        self.direction = action

        dx, dy = DIRECTION_VECTORS[self.direction]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        reward = 0
        if (new_head in self.snake) or \
           (new_head[0] < 0 or new_head[0] >= GRID_WIDTH) or \
           (new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.done = True
            reward = -1
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                reward = 1
                self.food = self.spawn_food()
            else:
                self.snake.pop()

        return self.get_state(), reward, self.done

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        dx = food_x - head_x
        dy = food_y - head_y

        # בונוס
        if self.bonus_active and self.bonus_food:
            bonus_x, bonus_y = self.bonus_food
            bdx = bonus_x - head_x
            bdy = bonus_y - head_y
            bonus_left = int(bdx < 0)
            bonus_up = int(bdy < 0)
        else:
            bonus_left = 0
            bonus_up = 0

        danger_left = self.check_collision(self.turn_left(self.direction))
        danger_right = self.check_collision(self.turn_right(self.direction))
        danger_up = self.check_collision('UP')
        danger_down = self.check_collision('DOWN')

        state = np.array([
            int(danger_left),
            int(danger_right),
            int(danger_up),
            int(danger_down),
            int(self.direction == 'LEFT'),
            int(self.direction == 'RIGHT'),
            int(self.direction == 'UP'),
            int(self.direction == 'DOWN'),
            int(dx < 0),
            int(dy < 0),
            len(self.snake) / (GRID_WIDTH * GRID_HEIGHT),
            bonus_left,
            bonus_up,
        ], dtype=np.float32)

        return state

    def is_opposite(self, dir1, dir2):
        return (dir1 == 'UP' and dir2 == 'DOWN') or \
               (dir1 == 'DOWN' and dir2 == 'UP') or \
               (dir1 == 'LEFT' and dir2 == 'RIGHT') or \
               (dir1 == 'RIGHT' and dir2 == 'LEFT')

    def check_collision(self, direction):
        dx, dy = DIRECTION_VECTORS[direction]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)
        return (
            new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT
        )

    def turn_left(self, direction):
        return {'UP': 'LEFT', 'LEFT': 'DOWN', 'DOWN': 'RIGHT', 'RIGHT': 'UP'}[direction]

    def turn_right(self, direction):
        return {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP'}[direction]
