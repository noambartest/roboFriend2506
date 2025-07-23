# ==== Imports ====
import tkinter as tk
from tkinter import messagebox
import threading
import serial
import time
import random
import pygame
from scipy.io import wavfile
import json
import os
import tempfile
import torch
import numpy as np
import pickle

# ==== Global Variables ====
collected_data = []   # Stores game examples for training
USE_AI = True         # AI mode flag

# ==== AI Model Definition ====
class SimonAI_MLP(torch.nn.Module):
    """Neural network for Simon AI decision making"""
    def __init__(self, input_dim):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU()
        )
        self.color_out = torch.nn.Linear(32, 12)   # 3 steps * 4 colors
        self.len_out = torch.nn.Linear(32, 3)      # step length (1/2/3)
        self.scale_out = torch.nn.Linear(32, 4)    # scale type
        self.comp_out = torch.nn.Linear(32, 1)     # complexity (regression)
        self.tempo_out = torch.nn.Linear(32, 1)    # tempo (regression)
    def forward(self, x):
        h = self.shared(x)
        colors = self.color_out(h)
        step_len = self.len_out(h)
        scale = self.scale_out(h)
        comp = self.comp_out(h)
        tempo = self.tempo_out(h)
        return colors, step_len, scale, comp, tempo

SCALE_MAP = {0: 'major', 1: 'minor', 2: 'pentatonic', 3: 'blues'}

# ==== Model Loading ====
mlp_model = SimonAI_MLP(input_dim=8)
mlp_model.load_state_dict(torch.load("simon/simon_mlp.pt"))
mlp_model.eval()

# ==== Serial Port Setup ====
ser = serial.Serial('COM5', 9600)
time.sleep(2)
ser.write(b"MODE SIMON\n")
time.sleep(0.2)

# ==== AI Decision Function ====
def get_ai_decision(sequence, reaction_times, round_num):
    """Decide next sequence step using the trained model"""
    seq_padded = (sequence + [0]*5)[:5]
    reacts_padded = (reaction_times + [1.0]*2)[:2]
    features = seq_padded + reacts_padded + [round_num/25.0]
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        colors_logits, len_logits, scale_logits, comp, tempo = mlp_model(x)
        colors_logits = colors_logits.view(3, 4)
        step_len = torch.softmax(len_logits, dim=1).argmax().item() + 1
        colors = []
        for i in range(3):
            p = torch.softmax(colors_logits[i], dim=0).numpy()
            colors.append(int(np.random.choice([0, 1, 2, 3], p=p)))
        scale = torch.softmax(scale_logits, dim=1).argmax().item()
        comp = float(comp.item())
        tempo = float(tempo.item())
    return colors, step_len, scale, comp, tempo

def save_example(features, y_colors, y_len, y_scale, y_comp, y_tempo):
    """Save a training example to collected_data for later training"""
    collected_data.append({
        'features': features,
        'y_colors': y_colors,
        'y_len': y_len,
        'y_scale': y_scale,
        'y_comp': y_comp,
        'y_tempo': y_tempo
    })

def back_to_robofriend():
    """Return to Robofriend menu"""
    root.destroy()
    # subprocess.Popen([sys.executable, "main_menu.py"])  # Uncomment if needed

# ==== MelodyAI: Handles all musical generation and adaptation ====
class MelodyAI:
    def __init__(self):
        self.base_frequencies = [261.63, 293.66, 329.63, 349.23]  # C, D, E, F
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'blues': [0, 3, 5, 6, 7, 10]
        }
        self.reset_composition()

    def reset_composition(self):
        """Reset all melody parameters for new game/composition"""
        self.current_scale = 'major'
        self.base_note = 60  # Middle C in MIDI
        self.melody_memory = []
        self.player_patterns = {}
        self.complexity = 0.0
        self.tempo_modifier = 1.0
        self.game_seed = random.randint(1, 10000)
        print(f"🎵 New composition started with seed: {self.game_seed}")

    def set_mood(self, round_num, success_rate=1.0):
        """Adapt musical style to game progress and success rate"""
        if success_rate > 0.8:
            self.current_scale = 'major'
            self.tempo_modifier = 1.0
        elif success_rate > 0.5:
            self.current_scale = 'pentatonic'
            self.tempo_modifier = 0.9
        else:
            self.current_scale = 'minor'
            self.tempo_modifier = 0.8
        self.complexity = min(round_num / 10.0, 1.0)

    def generate_note_for_color(self, color_index, position_in_sequence, sequence_so_far):
        """Create a musical note for a given color and position (deterministic)"""
        consistency_seed = self.game_seed + color_index + (position_in_sequence * 100)
        np.random.seed(consistency_seed)
        base_freq = self.base_frequencies[color_index]
        scale_notes = self.scales[self.current_scale]
        note_index = (color_index + position_in_sequence) % len(scale_notes)
        note_variation = scale_notes[note_index]
        octave_shift = (color_index % 3) - 1
        midi_note = self.base_note + note_variation + (octave_shift * 12)
        frequency = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        harmonies = []
        if self.complexity > 0.3:
            harmonies.append(frequency * 1.25)
        if self.complexity > 0.6:
            harmonies.append(frequency * 1.5)
        np.random.seed()
        return frequency, harmonies

    def generate_melody_sound(self, frequency, harmonies, duration=0.5):
        """Synthesize and play a musical sound using the selected note/harmonies"""
        sample_rate = 44100
        frames = int(duration * sample_rate)
        t = np.linspace(0, duration, frames)
        wave = 0.4 * np.sin(frequency * 2 * np.pi * t)
        for i, harm_freq in enumerate(harmonies):
            volume = 0.1 / (i + 1)
            wave += volume * np.sin(harm_freq * 2 * np.pi * t)
        attack_time = min(0.05, duration * 0.1)
        release_time = min(0.1, duration * 0.2)
        attack_frames = int(attack_time * sample_rate)
        release_frames = int(release_time * sample_rate)
        envelope = np.ones(frames)
        if attack_frames > 0:
            envelope[:attack_frames] = np.linspace(0, 1, attack_frames)
        if release_frames > 0:
            envelope[-release_frames:] = np.linspace(1, 0, release_frames)
        wave *= envelope
        if self.complexity > 0.8:
            noise = 0.005 * np.random.normal(0, 1, frames)
            wave += noise
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val * 0.8
        wave_16 = (wave * 32767).astype(np.int16)
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        try:
            wavfile.write(temp_filename, sample_rate, wave_16)
            sound = pygame.mixer.Sound(temp_filename)
            threading.Timer(2.0, lambda: self._cleanup_temp_file(temp_filename)).start()
            return sound
        except Exception as e:
            print(f"🚨 Error creating sound: {e}")
            return self._create_simple_beep(frequency, duration)

    def _create_simple_beep(self, frequency, duration):
        try:
            sample_rate = 44100
            frames = int(duration * sample_rate)
            t = np.linspace(0, duration, frames)
            wave = 0.3 * np.sin(frequency * 2 * np.pi * t)
            envelope = np.exp(-2 * t / duration)
            wave *= envelope
            wave_16 = (wave * 32767 * 0.5).astype(np.int16)
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_filename = temp_file.name
            temp_file.close()
            wavfile.write(temp_filename, sample_rate, wave_16)
            sound = pygame.mixer.Sound(temp_filename)
            threading.Timer(1.0, lambda: self._cleanup_temp_file(temp_filename)).start()
            return sound
        except:
            return pygame.mixer.Sound(buffer=np.zeros(1000, dtype=np.int16))

    def _cleanup_temp_file(self, filename):
        try:
            if os.path.exists(filename):
                os.unlink(filename)
        except:
            pass

    def learn_from_player(self, sequence, reaction_times):
        avg_reaction = np.mean(reaction_times) if reaction_times else 1.0
        if avg_reaction < 0.5:
            self.tempo_modifier = min(self.tempo_modifier * 1.1, 1.5)
        elif avg_reaction > 1.0:
            self.tempo_modifier = max(self.tempo_modifier * 0.9, 0.6)
        pattern = tuple(sequence)
        if pattern not in self.player_patterns:
            self.player_patterns[pattern] = []
        self.player_patterns[pattern].append(avg_reaction)

    def save_composition(self, sequence, round_num):
        composition = {
            'sequence': [int(x) for x in sequence],
            'scale': self.current_scale,
            'round': round_num,
            'timestamp': time.time()
        }
        if not os.path.exists('compositions'):
            os.makedirs('compositions')
        filename = f"compositions/simon_composition_round_{round_num}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(composition, f, indent=2)

# ==== Initialize Music AI ====
melody_ai = MelodyAI()

# ==== Pygame / Sounds Setup ====
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
error_snd = pygame.mixer.Sound("sounds/error.wav")
win_snd = pygame.mixer.Sound("sounds/win.wav")

# ==== GUI Setup ====
root = tk.Tk()
root.title("🎵 AI Musical Simon Says ")
root.geometry("600x400")
root.configure(bg="#1e1e2f")
frame_intro = tk.Frame(root, bg="#1e1e2f")
frame_game = tk.Frame(root, bg="#1e1e2f")
status_label = tk.Label(frame_game, text="", font=("Segoe UI", 18), fg="white", bg="#1e1e2f")
status_label.pack(pady=30)
music_info_label = tk.Label(frame_game, text="", font=("Segoe UI", 12), fg="#4dd0e1", bg="#1e1e2f")
music_info_label.pack(pady=10)
restart_button = tk.Button(
    frame_game, text="🔁 Restart", font=("Segoe UI", 14, "bold"),
    bg="#4a90e2", fg="white", activebackground="#357ABD", relief="flat", padx=20, pady=10,
    command=lambda: show_intro()
)
restart_button.pack(pady=10)
restart_button.pack_forget()
back_button = tk.Button(
    frame_game, text="⬅️ Back to Robofriend", font=("Segoe UI", 13, "bold"),
    bg="#616161", fg="white", activebackground="#4f4f4f", relief="flat", padx=20, pady=8,
    command=lambda: back_to_robofriend()
)
back_button.pack_forget()

# ==== Utility Functions ====
def send_play_command(idx):
    """Send play command to Arduino"""
    ser.write(f"PLAY {idx}\n".encode())

def play_musical_sound(idx, position, sequence_so_far, duration=0.5):
    """Play a new musical note generated by the AI"""
    try:
        frequency, harmonies = melody_ai.generate_note_for_color(idx, position, sequence_so_far)
        sound = melody_ai.generate_melody_sound(frequency, harmonies, duration)
        music_info = f"🎼 Scale: {melody_ai.current_scale.title()} | Note: {frequency:.1f}Hz"
        if harmonies:
            music_info += f" + {len(harmonies)} Harmonies"
        music_info_label.config(text=music_info)
        print(f"🎵 Playing: Color {idx}, Position {position}, Freq: {frequency:.1f}Hz")
        if sound:
            sound.play()
            time.sleep(duration)
            sound.stop()
        else:
            print(f"⚠️ No sound created for color {idx}")
    except Exception as e:
        print(f"🚨 Error in play_musical_sound: {e}")
        frequency = melody_ai.base_frequencies[idx]
        music_info_label.config(text=f"🎼 Fallback: {frequency:.1f}Hz")

def play_sequence(seq, round_num, success_rate=1.0):
    """Play a sequence with dynamic music"""
    melody_ai.set_mood(round_num, success_rate)
    for i, idx in enumerate(seq):
        send_play_command(idx)
        duration = 0.5 * melody_ai.tempo_modifier
        play_musical_sound(idx, i, seq[:i], duration)
        time.sleep(0.1)
    melody_ai.save_composition(seq, round_num)

def get_player_input(expected_sequence):
    """Handle player input (with button events from Arduino)"""
    inputs = []
    reaction_times = []
    for i, expected_idx in enumerate(expected_sequence):
        start_time = time.time()
        while True:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                if line.startswith("BTN"):
                    reaction_time = time.time() - start_time
                    reaction_times.append(reaction_time)
                    idx = int(line.split()[1])
                    play_musical_sound(idx, i, expected_sequence[:i])
                    if idx != expected_idx:
                        return None, reaction_times
                    inputs.append(idx)
                    break
    melody_ai.learn_from_player(expected_sequence, reaction_times)
    return inputs, reaction_times

def set_status_animated(text, color="white"):
    """Animated status display in the GUI"""
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

# ==== Main Game Logic ====
def start_game(rounds):
    frame_intro.pack_forget()
    frame_game.pack()
    restart_button.pack_forget()
    back_button.pack_forget()
    sequence = []
    success_count = 0
    start_game.reaction_times_history = []

    def game_loop(round_num):
        nonlocal success_count
        if round_num > rounds:
            if win_snd:
                win_snd.play()
            set_status_animated(f"🎉 Game Complete! Musical Score: {success_count}/{rounds}", "#00ff99")
            music_info_label.config(text=f"🎼 Final Composition saved! Style: {melody_ai.current_scale.title()}")
            restart_button.pack()
            back_button.pack(pady=8)
            with open("simon_data.pkl", "wb") as f:
                pickle.dump(collected_data, f)
            print(f"✅ Saved {len(collected_data)} examples to simon_data.pkl")
            return

        set_status_animated(f"♪ MUSICAL ROUND {round_num} ♪", "#4dd0e1")
        root.after(950, lambda: continue_round(round_num))

    def continue_round(round_num):
        nonlocal success_count
        if not hasattr(start_game, "reaction_times_history"):
            start_game.reaction_times_history = []
        reaction_times = start_game.reaction_times_history[-1] if start_game.reaction_times_history else []
        success_rate = success_count / max(round_num - 1, 1) if round_num > 1 else 1.0

        if not USE_AI:
            next_color = random.randint(0, 3)
            sequence.append(next_color)
        else:
            colors, step_len, scale, comp, tempo = get_ai_decision(sequence, reaction_times, round_num)
            sequence.extend(colors[:step_len])
            melody_ai.current_scale = SCALE_MAP[scale]
            melody_ai.complexity = comp
            melody_ai.tempo_modifier = tempo

        play_sequence(sequence, round_num, success_rate)
        set_status_animated("🎵 Your turn to play the melody... 🎵", "#ffcc00")

        def wait_for_input():
            nonlocal success_count
            player_input, reaction_times = get_player_input(sequence)
            if not hasattr(start_game, "reaction_times_history"):
                start_game.reaction_times_history = []
            start_game.reaction_times_history.append(reaction_times)
            if player_input is None:
                set_status_animated("🎵 Wrong note! Game over ❌", "#ff5252")
                if error_snd:
                    error_snd.play()
                    time.sleep(error_snd.get_length())
                music_info_label.config(text="🎼 Try again to create a new composition!")
                restart_button.pack()
                back_button.pack(pady=8)
            else:
                # --- Save a training example for each round ---
                seq_padded = (sequence + [0] * 5)[:5]
                reacts_padded = (reaction_times + [1.0] * 2)[:2]
                features = seq_padded + reacts_padded + [round_num / 25.0]
                save_example(
                    features=features,
                    y_colors=sequence[-3:] if len(sequence) >= 3 else sequence + [0] * (3 - len(sequence)),
                    y_len=min(2, len(sequence) - 1),
                    y_scale=list(SCALE_MAP.keys())[list(SCALE_MAP.values()).index(melody_ai.current_scale)],
                    y_comp=melody_ai.complexity,
                    y_tempo=melody_ai.tempo_modifier
                )
                success_count += 1
                avg_reaction = np.mean(reaction_times)
                set_status_animated("🎵 Perfect harmony! ✅", "#00ff99")
                music_info_label.config(text=f"🎼 Reaction time: {avg_reaction:.2f}s | AI adapting...")
                root.after(1500, lambda: game_loop(round_num + 1))

        threading.Thread(target=wait_for_input, daemon=True).start()

    game_loop(1)

# ==== GUI Intro Screen ====
def show_intro():
    frame_game.pack_forget()
    frame_intro.pack()
    for widget in frame_intro.winfo_children():
        widget.destroy()
    intro_label = tk.Label(frame_intro, text="🎼 AI Musical Simon Says 🎼",
                           font=("Segoe UI", 20, "bold"), fg="#4dd0e1", bg="#1e1e2f")
    intro_label.pack(pady=20)
    desc_label = tk.Label(frame_intro,
                          text="Each game creates a unique musical composition!\nThe AI adapts the melody based on your performance.",
                          font=("Segoe UI", 12), fg="white", bg="#1e1e2f", justify='center')
    desc_label.pack(pady=10)
    rounds_label = tk.Label(frame_intro, text="How many rounds would you like to compose?",
                            font=("Segoe UI", 16), fg="white", bg="#1e1e2f")
    rounds_label.pack(pady=20)
    entry = tk.Entry(frame_intro, font=("Segoe UI", 14), justify='center')
    entry.pack(pady=10)
    entry.focus()
    def on_select_rounds():
        try:
            rounds = int(entry.get())
            if rounds <= 0:
                raise ValueError
            melody_ai.reset_composition()
            print(f"🎮 Starting new game with {rounds} rounds")
            threading.Thread(target=start_game, args=(rounds,), daemon=True).start()
        except ValueError:
            messagebox.showerror("Error", "Please enter a positive number")
    start_button = tk.Button(
        frame_intro, text="🎵 Start Composing! 🚀", font=("Segoe UI", 14, "bold"), command=on_select_rounds,
        bg="#28a745", fg="white", activebackground="#218838", relief="flat", padx=20, pady=10
    )
    start_button.pack(pady=20)

# ==== Mainloop ====
show_intro()
root.mainloop()
