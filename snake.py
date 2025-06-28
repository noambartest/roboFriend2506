import os
import tkinter as tk
import serial
import threading
import random
import time

ser = serial.Serial('COM5', 9600)
time.sleep(2)
ser.write(b"MODE SNAKE\n")
time.sleep(0.2)

GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20

DIRECTIONS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
}

class SnakeGame:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=GRID_WIDTH * GRID_SIZE, height=GRID_HEIGHT * GRID_SIZE, bg='black')
        self.canvas.pack()
        self.restart_button = None
        self.high_score = self.load_high_score()

        self.joystick_pressed = False
        self.joystick_restart_allowed = True
        self.after_id = None
        self.direction_changed = False
        self.deadzone = 100
        self.game_active = True

        self.pulse_radius = 0
        self.pulse_direction = 1
        self.just_ate = False

        self.read_joystick_thread = threading.Thread(target=self.read_joystick)
        self.read_joystick_thread.daemon = True
        self.read_joystick_thread.start()

        self.setup_game()
        self.update()

    def setup_game(self):
        self.snake = [(5, 5)]
        self.food = self.spawn_food()
        self.direction = 'RIGHT'
        self.score = 0
        self.speed = 175
        self.direction_changed = False
        self.canvas.delete("all")
        self.game_active = True

        self.walls = []
        self.progress_counter = 0
        self.last_upgrade = None
        self.bonus_food = None
        self.bonus_active = False
        self.bonus_timer = 0
        self.just_ate = False
        self.pulse_radius = 0
        self.pulse_direction = 1

        if self.restart_button:
            self.restart_button.destroy()
            self.restart_button = None
        if hasattr(self, 'back_button') and self.back_button:
            self.back_button.destroy()
            self.back_button = None
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def draw(self):
        self.canvas.delete("all")
        for i, (x, y) in enumerate(self.snake):
            color = "green"
            if i == 0:
                color = "lightgreen"
            elif i == len(self.snake) - 1 and self.just_ate:
                color = "#0f0"
            self.canvas.create_rectangle(x * GRID_SIZE, y * GRID_SIZE,
                                         (x + 1) * GRID_SIZE, (y + 1) * GRID_SIZE, fill=color)

        fx, fy = self.food
        pulse = self.pulse_radius
        self.canvas.create_oval(fx * GRID_SIZE - pulse, fy * GRID_SIZE - pulse,
                                (fx + 1) * GRID_SIZE + pulse, (fy + 1) * GRID_SIZE + pulse,
                                fill='red')

        for x, y in self.walls:
            self.canvas.create_rectangle(x * GRID_SIZE, y * GRID_SIZE,
                                         (x + 1) * GRID_SIZE, (y + 1) * GRID_SIZE, fill='gray')

        if self.bonus_active and self.bonus_food:
            bx, by = self.bonus_food
            self.canvas.create_oval(bx * GRID_SIZE, by * GRID_SIZE,
                                    (bx + 1) * GRID_SIZE, (by + 1) * GRID_SIZE, fill='blue')

        self.canvas.create_text(10, 10, anchor="nw", fill="white", text=f"Score: {self.score}")
        self.canvas.create_text(10, 30, anchor="nw", fill="yellow", text=f"High Score: {self.high_score}")

    def is_opposite_direction(self, current, new):
        return (current == 'UP' and new == 'DOWN') or \
               (current == 'DOWN' and new == 'UP') or \
               (current == 'LEFT' and new == 'RIGHT') or \
               (current == 'RIGHT' and new == 'LEFT')

    def load_high_score(self):
        if not os.path.exists("highscore.txt"):
            with open("highscore.txt", "w") as f:
                f.write("0")
            return 0
        with open("highscore.txt", "r") as f:
            return int(f.read().strip())

    def back_to_menu(self):
        self.root.destroy()


    def update(self):
        if not self.game_active:
            if not self.restart_button:
                self.canvas.create_text(GRID_WIDTH * GRID_SIZE // 2, GRID_HEIGHT * GRID_SIZE // 2 - 20,
                                        text="GAME OVER", fill="white", font=('Arial', 24))
                self.restart_button = tk.Button(self.root, text="RESTART", font=('Arial', 14),
                                                command=self.restart_game)
                self.canvas.create_window(GRID_WIDTH * GRID_SIZE // 2, GRID_HEIGHT * GRID_SIZE // 2 + 20,
                                          window=self.restart_button)

            if self.joystick_pressed and self.joystick_restart_allowed:
                self.joystick_restart_allowed = False
                self.restart_game()

            if not self.joystick_pressed:
                self.joystick_restart_allowed = True

            self.after_id = self.root.after(100, self.update)
            return

        if self.snake_has_collided():
            self.game_active = False
            if self.restart_button:
                self.restart_button.destroy()
            self.restart_button = tk.Button(self.root, text="🔁 Restart", font=('Arial', 14), command=self.restart_game)
            self.canvas.create_window(GRID_WIDTH * GRID_SIZE // 2, GRID_HEIGHT * GRID_SIZE // 2 + 20,
                                      window=self.restart_button)
            self.back_button = tk.Button(self.root, text="⬅ Back to Robofriend", font=('Arial', 12),
                                         command=self.back_to_menu)
            self.canvas.create_window(GRID_WIDTH * GRID_SIZE // 2, GRID_HEIGHT * GRID_SIZE // 2 + 60,
                                      window=self.back_button)
            self.after_id = self.root.after(100, self.update)
            return

        head = self.snake[0]
        dx, dy = DIRECTIONS[self.direction]
        new_head = (head[0] + dx, head[1] + dy)
        self.snake = [new_head] + self.snake

        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            self.just_ate = True
            if self.score > self.high_score:
                self.high_score = self.score
                with open("highscore.txt", "w") as f:
                    f.write(str(self.high_score))

            if self.score % 2 == 0:
                self.add_wall()

            if self.score % 4 == 0:
                if self.speed > 50:
                    self.speed -= 5

        else:
            self.snake.pop()
            self.just_ate = False

        if self.bonus_active and self.bonus_food:
            if self.snake[0] == self.bonus_food:
                self.score += 5
                if self.score > self.high_score:
                    self.high_score = self.score
                    with open("highscore.txt", "w") as f:
                        f.write(str(self.high_score))
                self.speed = min(self.speed + 20, 200)
                self.bonus_active = False
                self.bonus_food = None
            else:
                self.bonus_timer -= 1
                if self.bonus_timer <= 0:
                    self.bonus_active = False
                    self.bonus_food = None
        else:
            if random.randint(1, 150) == 1:
                self.spawn_bonus_food()

        self.pulse_radius += self.pulse_direction
        if self.pulse_radius >= 3 or self.pulse_radius <= 0:
            self.pulse_direction *= -1

        self.direction_changed = False
        self.draw()
        self.after_id = self.root.after(self.speed, self.update)

    def snake_has_collided(self):
        head = self.snake[0]
        return (
                head in self.snake[1:] or
                head in self.walls or
                head[0] < 0 or head[0] >= GRID_WIDTH or
                head[1] < 0 or head[1] >= GRID_HEIGHT
        )

    def spawn_food(self):
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in self.snake:
                return pos

    def read_joystick(self):
        while True:
            try:
                line = ser.readline().decode().strip()
                x_str, y_str, pressed_str = line.split(',')
                x, y = int(x_str), int(y_str)
                pressed = (pressed_str == "1")

                if self.game_active:
                    if not self.direction_changed:
                        if x < 512 - self.deadzone and not self.is_opposite_direction(self.direction, 'LEFT'):
                            self.direction = 'LEFT'
                            self.direction_changed = True
                        elif x > 512 + self.deadzone and not self.is_opposite_direction(self.direction, 'RIGHT'):
                            self.direction = 'RIGHT'
                            self.direction_changed = True
                        elif y < 512 - self.deadzone and not self.is_opposite_direction(self.direction, 'UP'):
                            self.direction = 'UP'
                            self.direction_changed = True
                        elif y > 512 + self.deadzone and not self.is_opposite_direction(self.direction, 'DOWN'):
                            self.direction = 'DOWN'
                            self.direction_changed = True

                self.joystick_pressed = pressed
            except:
                continue

    def restart_game(self):
        self.setup_game()
        #self.update()

    def add_wall(self):
        while True:
            x = random.randint(1, GRID_WIDTH - 2)
            y = random.randint(1, GRID_HEIGHT - 2)
            pos = (x, y)
            if pos not in self.snake and pos != self.food and pos not in self.walls:
                self.walls.append(pos)
                break

    def spawn_bonus_food(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            pos = (x, y)
            if pos not in self.snake and pos != self.food and pos not in self.walls:
                self.bonus_food = pos
                self.bonus_active = True
                self.bonus_timer = 50
                break

root = tk.Tk()
root.title("Arduino Snake Game")
game = SnakeGame(root)
root.mainloop()