import tkinter as tk
from tkinter import messagebox
import threading
import serial
import time
import random
import pygame

ser = serial.Serial('COM5', 9600)
time.sleep(2)
ser.write(b"MODE SIMON\n")
time.sleep(0.2)

sound_files = ["sounds/DO.wav", "sounds/RE.wav", "sounds/MI.wav", "sounds/FA.wav"]
pygame.mixer.init()
sounds = [pygame.mixer.Sound(f) for f in sound_files]
error_snd = pygame.mixer.Sound("sounds/error.wav")
win_snd = pygame.mixer.Sound("sounds/win.wav")

# --- GUI ראשי ---
root = tk.Tk()
root.title("🎵 Simon Says ")
root.geometry("500x320")
root.configure(bg="#1e1e2f")

frame_intro = tk.Frame(root, bg="#1e1e2f")
frame_game = tk.Frame(root, bg="#1e1e2f")

status_label = tk.Label(frame_game, text="", font=("Segoe UI", 18), fg="white", bg="#1e1e2f")
status_label.pack(pady=30)

restart_button = tk.Button(
    frame_game, text="🔁 Restart", font=("Segoe UI", 14, "bold"),
    bg="#4a90e2", fg="white", activebackground="#357ABD", relief="flat", padx=20, pady=10,
    command=lambda: show_intro()
)
restart_button.pack(pady=10)
restart_button.pack_forget()


def send_play_command(idx):
    ser.write(f"PLAY {idx}\n".encode())

def play_sound(idx, duration=0.5):
    snd = sounds[idx]
    snd.play()
    time.sleep(duration)
    snd.stop()

def play_sequence(seq):
    for idx in seq:
        send_play_command(idx)
        play_sound(idx)
        time.sleep(0.1)

def get_player_input(expected_sequence):
    inputs = []
    for expected_idx in expected_sequence:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                if line.startswith("BTN"):
                    idx = int(line.split()[1])
                    play_sound(idx)
                    if idx != expected_idx:
                        return None
                    inputs.append(idx)
                    break
    return inputs

def set_status_animated(text, color="white"):
    status_label.config(fg=color)
    status_label["text"] = ""
    i = 0
    def animate():
        nonlocal i
        if i < len(text):
            status_label["text"] += text[i]
            i += 1
            root.after(50, animate)
    animate()

# --- לוגיקת משחק ---
def start_game(rounds):
    frame_intro.pack_forget()
    frame_game.pack()
    restart_button.pack_forget()
    sequence = []

    def game_loop(round_num):
        if round_num > rounds:
            win_snd.play()
            #time.sleep(win_snd.get_length())
            set_status_animated(" You won the game!🎉", "#00ff99")
            restart_button.pack()
            return
        set_status_animated(f" ROUND {round_num}  📣", "#4dd0e1")
        root.after(950, lambda: continue_round(round_num))

    def continue_round(round_num):
        next_color = random.randint(0, 3)
        sequence.append(next_color)
        play_sequence(sequence)
        set_status_animated("waiting for input...⏳ ", "#ffcc00")

        def wait_for_input():
            player_input = get_player_input(sequence)
            if player_input is None:
                set_status_animated("Wrong! game over ❌", "#ff5252")
                error_snd.play()
                time.sleep(error_snd.get_length())
                restart_button.pack()
            else:
                set_status_animated("Success! let's continue ✅ ", "#00ff99")
                root.after(1500, lambda: game_loop(round_num + 1))

        threading.Thread(target=wait_for_input, daemon=True).start()

    game_loop(1)

# --- מסך פתיחה ---
def show_intro():
    frame_game.pack_forget()
    frame_intro.pack()
    for widget in frame_intro.winfo_children():
        widget.destroy()

    intro_label = tk.Label(frame_intro, text="How many rounds you would like to play?", font=("Segoe UI", 16), fg="white", bg="#1e1e2f")
    intro_label.pack(pady=30)

    entry = tk.Entry(frame_intro, font=("Segoe UI", 14), justify='center')
    entry.pack(pady=10)
    entry.focus()

    def on_select_rounds():
        try:
            rounds = int(entry.get())
            if rounds <= 0:
                raise ValueError
            threading.Thread(target=start_game, args=(rounds,), daemon=True).start()
        except ValueError:
            messagebox.showerror("error", "please enter a positive number")

    start_button = tk.Button(
        frame_intro, text="Start to Play 🚀 ", font=("Segoe UI", 14, "bold"), command=on_select_rounds,
        bg="#28a745", fg="white", activebackground="#218838", relief="flat", padx=20, pady=10
    )
    start_button.pack(pady=20)


show_intro()
root.mainloop()
