import tkinter as tk
from tkinter import messagebox
import threading
import serial
import time
import random
import pygame
import numpy as np
from scipy import signal
from scipy.io import wavfile
import json
import os
import tempfile
import torch


class SimonMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# טוענים את המודל המאומן
mlp_model = SimonMLP(input_dim=8, output_dim=4)
mlp_model.load_state_dict(torch.load("simon_mlp.pt"))
mlp_model.eval()


# אותה הגדרה של Serial כמו קודם
ser = serial.Serial('COM5', 9600)
time.sleep(2)
ser.write(b"MODE SIMON\n")
time.sleep(0.2)

def get_next_color_ai(sequence, reaction_times, round_num):
    # דואג לריפוד רשימות קצרות
    seq_padded = (sequence + [0]*5)[:5]
    reacts_padded = (reaction_times + [1.0]*2)[:2]
    features = seq_padded + reacts_padded + [round_num/25.0]
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = mlp_model(x)
        probs = torch.softmax(output, dim=1).numpy().flatten()
        next_color = np.random.choice([0, 1, 2, 3], p=probs)
    return next_color


# מחלקה ליצירת מוזיקה דינמית
class MelodyAI:
    def __init__(self):
        self.base_frequencies = [261.63, 293.66, 329.63, 349.23]  # C, D, E, F
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'blues': [0, 3, 5, 6, 7, 10]
        }
        self.reset_composition()  # איפוס למשחק חדש

    def reset_composition(self):
        """איפוס מלא למשחק חדש"""
        self.current_scale = 'major'
        self.base_note = 60  # Middle C in MIDI
        self.melody_memory = []  # זוכר מנגינות קודמות
        self.player_patterns = {}  # לומד דפוסי שחקן
        self.complexity = 0.0
        self.tempo_modifier = 1.0
        self.game_seed = random.randint(1, 10000)  # זרע ייחודי לכל משחק
        print(f"🎵 New composition started with seed: {self.game_seed}")

    def set_mood(self, round_num, success_rate=1.0):
        """מגדיר מצב רוח בהתאם לרמה והצלחות"""
        if success_rate > 0.8:
            self.current_scale = 'major'
            self.tempo_modifier = 1.0
        elif success_rate > 0.5:
            self.current_scale = 'pentatonic'
            self.tempo_modifier = 0.9
        else:
            self.current_scale = 'minor'
            self.tempo_modifier = 0.8

        # ככל שהרמה גבוהה יותר, המוזיקה מורכבת יותר
        self.complexity = min(round_num / 10.0, 1.0)

    def generate_note_for_color(self, color_index, position_in_sequence, sequence_so_far):
        """יוצר תו מותאם לצבע, למיקום ברצף ולהיסטוריה - עקבי בתוך המשחק"""

        # השתמש בזרע קבוע + מיקום כדי להבטיח עקביות בתוך המשחק
        consistency_seed = self.game_seed + color_index + (position_in_sequence * 100)
        np.random.seed(consistency_seed)

        # בסיס: התו הקבוע של הצבע עם וריאציה קלה
        base_freq = self.base_frequencies[color_index]

        # הוסף וריאציה עקבית בהתאם למיקום ברצף
        scale_notes = self.scales[self.current_scale]

        # חישוב עקבי של התו
        note_index = (color_index + position_in_sequence) % len(scale_notes)
        note_variation = scale_notes[note_index]

        # אוקטבה קבועה לפי צבע
        octave_shift = (color_index % 3) - 1  # -1, 0, 1 בהתאם לצבע

        # חישב תדירות חדשה בצורה עקבית
        midi_note = self.base_note + note_variation + (octave_shift * 12)
        frequency = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

        # הרמוניות עקביות אם הרמה מספיק גבוהה
        harmonies = []
        if self.complexity > 0.3:
            harmonies.append(frequency * 1.25)  # רביעית
        if self.complexity > 0.6:
            harmonies.append(frequency * 1.5)  # קווינטה

        # איפוס ה-seed כדי לא להשפיע על דברים אחרים
        np.random.seed()

        return frequency, harmonies

    def generate_melody_sound(self, frequency, harmonies, duration=0.5):
        """יוצר צליל מוזיקלי עם הרמוניות ושומר כקובץ WAV זמני"""
        sample_rate = 44100
        frames = int(duration * sample_rate)

        # גל עיקרי - סינוס נקי
        t = np.linspace(0, duration, frames)
        wave = 0.4 * np.sin(frequency * 2 * np.pi * t)

        # הוסף הרמוניות בעוצמה נמוכה יותר
        for i, harm_freq in enumerate(harmonies):
            volume = 0.1 / (i + 1)  # הרמוניות יותר חלשות
            wave += volume * np.sin(harm_freq * 2 * np.pi * t)

        # envelope חלק לתחילה וסוף
        attack_time = min(0.05, duration * 0.1)  # 5% מהזמן או 50ms
        release_time = min(0.1, duration * 0.2)  # 10% מהזמן או 100ms

        attack_frames = int(attack_time * sample_rate)
        release_frames = int(release_time * sample_rate)

        # יצירת envelope
        envelope = np.ones(frames)

        # התחלה חלקה
        if attack_frames > 0:
            envelope[:attack_frames] = np.linspace(0, 1, attack_frames)

        # סוף חלק
        if release_frames > 0:
            envelope[-release_frames:] = np.linspace(1, 0, release_frames)

        wave *= envelope

        # רק אם הרמת מורכבות גבוהה מאוד - הוסף רעש מינימלי
        if self.complexity > 0.8:
            noise = 0.005 * np.random.normal(0, 1, frames)  # רעש מינימלי
            wave += noise

        # נרמל ופליטה - בטוח שלא נחרוג
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val * 0.8  # השאר מרווח בטיחות

        wave_16 = (wave * 32767).astype(np.int16)

        # צור קובץ WAV זמני
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_filename = temp_file.name
        temp_file.close()

        try:
            # שמור כקובץ WAV
            wavfile.write(temp_filename, sample_rate, wave_16)

            # טען חזרה כ-pygame Sound
            sound = pygame.mixer.Sound(temp_filename)

            # נקה קובץ זמני (לאחר עיכוב קצר)
            threading.Timer(2.0, lambda: self._cleanup_temp_file(temp_filename)).start()

            return sound

        except Exception as e:
            print(f"🚨 Error creating sound: {e}")
            # במקרה של שגיאה, החזר צליל פשוט
            return self._create_simple_beep(frequency, duration)

    def _create_simple_beep(self, frequency, duration):
        """יוצר צליל פשוט במקרה של שגיאה"""
        try:
            sample_rate = 44100
            frames = int(duration * sample_rate)
            t = np.linspace(0, duration, frames)
            wave = 0.3 * np.sin(frequency * 2 * np.pi * t)

            # envelope פשוט
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
            # במקרה הגרוע ביותר - שקט
            return pygame.mixer.Sound(buffer=np.zeros(1000, dtype=np.int16))

    def _cleanup_temp_file(self, filename):
        """מנקה קובץ זמני"""
        try:
            if os.path.exists(filename):
                os.unlink(filename)
        except:
            pass  # אם לא מצליח למחוק, לא נורא

    def learn_from_player(self, sequence, reaction_times):
        """לומד מהתנהגות השחקן ומתאים מוזיקה"""
        avg_reaction = np.mean(reaction_times) if reaction_times else 1.0

        # שחקן מהיר = מוזיקה מהירה יותר
        if avg_reaction < 0.5:
            self.tempo_modifier = min(self.tempo_modifier * 1.1, 1.5)
        elif avg_reaction > 1.0:
            self.tempo_modifier = max(self.tempo_modifier * 0.9, 0.6)

        # שמור דפוסים
        pattern = tuple(sequence)
        if pattern not in self.player_patterns:
            self.player_patterns[pattern] = []
        self.player_patterns[pattern].append(avg_reaction)

    def save_composition(self, sequence, round_num):
        """שומר את הקומפוזיציה שנוצרה"""
        composition = {
            'sequence': [int(x) for x in sequence],
            'scale': self.current_scale,
            'round': round_num,
            'timestamp': time.time()
        }

        # שמור לקובץ
        if not os.path.exists('compositions'):
            os.makedirs('compositions')

        filename = f"compositions/simon_composition_round_{round_num}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(composition, f, indent=2)


# יצירת מופע של ה-AI המוזיקלי
melody_ai = MelodyAI()

# אתחול pygame
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
error_snd = pygame.mixer.Sound("../sounds/error.wav")
win_snd = pygame.mixer.Sound("../sounds/win.wav")

# GUI זהה לקודם
root = tk.Tk()
root.title("🎵 AI Musical Simon Says ")
root.geometry("600x400")
root.configure(bg="#1e1e2f")

frame_intro = tk.Frame(root, bg="#1e1e2f")
frame_game = tk.Frame(root, bg="#1e1e2f")

status_label = tk.Label(frame_game, text="", font=("Segoe UI", 18), fg="white", bg="#1e1e2f")
status_label.pack(pady=30)

# תווית חדשה להצגת מידע מוזיקלי
music_info_label = tk.Label(frame_game, text="", font=("Segoe UI", 12), fg="#4dd0e1", bg="#1e1e2f")
music_info_label.pack(pady=10)

restart_button = tk.Button(
    frame_game, text="🔁 Restart", font=("Segoe UI", 14, "bold"),
    bg="#4a90e2", fg="white", activebackground="#357ABD", relief="flat", padx=20, pady=10,
    command=lambda: show_intro()
)
restart_button.pack(pady=10)
restart_button.pack_forget()


def send_play_command(idx):
    ser.write(f"PLAY {idx}\n".encode())


def play_musical_sound(idx, position, sequence_so_far, duration=0.5):
    """משחק צליל מוזיקלי חדש שנוצר ע"י ה-AI"""
    try:
        frequency, harmonies = melody_ai.generate_note_for_color(idx, position, sequence_so_far)
        sound = melody_ai.generate_melody_sound(frequency, harmonies, duration)

        # עדכון מידע מוזיקלי בGUI
        music_info = f"🎼 Scale: {melody_ai.current_scale.title()} | Note: {frequency:.1f}Hz"
        if harmonies:
            music_info += f" + {len(harmonies)} Harmonies"
        music_info_label.config(text=music_info)

        print(f"🎵 Playing: Color {idx}, Position {position}, Freq: {frequency:.1f}Hz")

        # וודא שהצליל מתנגן
        if sound:
            sound.play()
            time.sleep(duration)
            sound.stop()
        else:
            print(f"⚠️ No sound created for color {idx}")

    except Exception as e:
        print(f"🚨 Error in play_musical_sound: {e}")
        # נגן צליל פשוט במקרה של שגיאה
        frequency = melody_ai.base_frequencies[idx]
        music_info_label.config(text=f"🎼 Fallback: {frequency:.1f}Hz")
        # כאן תוכל להוסיף צליל חלופי


def play_sequence(seq, round_num, success_rate=1.0):
    """משחק רצף עם מוזיקה דינמית"""
    melody_ai.set_mood(round_num, success_rate)

    for i, idx in enumerate(seq):
        send_play_command(idx)
        duration = 0.5 * melody_ai.tempo_modifier
        play_musical_sound(idx, i, seq[:i], duration)
        time.sleep(0.1)

    # שמור קומפוזיציה
    melody_ai.save_composition(seq, round_num)


def get_player_input(expected_sequence):
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
                    # השמע את התו שהשחקן לחץ עליו
                    play_musical_sound(idx, i, expected_sequence[:i])

                    if idx != expected_idx:
                        return None, reaction_times
                    inputs.append(idx)
                    break

    # למד מהתנהגות השחקן
    melody_ai.learn_from_player(expected_sequence, reaction_times)
    return inputs, reaction_times


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


# לוגיקת משחק מעודכנת
def start_game(rounds):
    frame_intro.pack_forget()
    frame_game.pack()
    restart_button.pack_forget()
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
            return

        set_status_animated(f"♪ MUSICAL ROUND {round_num} ♪", "#4dd0e1")
        root.after(950, lambda: continue_round(round_num))

    def continue_round(round_num):
        nonlocal success_count
        if not hasattr(start_game, "reaction_times_history"):
            start_game.reaction_times_history = []
        # בכל סיבוב, מקבלים את זמני התגובה מהסיבוב הקודם (אם קיים)
        reaction_times = start_game.reaction_times_history[-1] if start_game.reaction_times_history else []
        next_color = get_next_color_ai(sequence, reaction_times, round_num)
        sequence.append(next_color)
        success_rate = success_count / max(round_num - 1, 1) if round_num > 1 else 1.0

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
            else:
                success_count += 1
                avg_reaction = np.mean(reaction_times)
                set_status_animated("🎵 Perfect harmony! ✅", "#00ff99")
                music_info_label.config(text=f"🎼 Reaction time: {avg_reaction:.2f}s | AI adapting...")
                root.after(1500, lambda: game_loop(round_num + 1))

        threading.Thread(target=wait_for_input, daemon=True).start()

    game_loop(1)


# מסך פתיחה מעודכן
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
            # איפוס ה-AI לקומפוזיציה חדשה - זה החשוב!
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


show_intro()
root.mainloop()

error_snd = pygame.mixer.Sound("../sounds/error.wav")
win_snd = pygame.mixer.Sound("../sounds/win.wav")