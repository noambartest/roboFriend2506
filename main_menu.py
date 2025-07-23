import subprocess
import sys
import serial
import time
import ttkbootstrap as tb
from ttkbootstrap.constants import *

ARDUINO_PORT = "COM5"
BAUD_RATE = 9600

def send_mode_to_arduino(mode):
    try:
        print(f"שולח MODE {mode.upper()} ל-{ARDUINO_PORT}...")
        with serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2)
            ser.write(f"MODE {mode.upper()}\n".encode())
            print("נשלח ✔️")
            time.sleep(0.2)
    except serial.SerialException as e:
        print(f"❌ שגיאה בתקשורת עם Arduino: {e}")

def launch_simon():
    send_mode_to_arduino("SIMON")
    time.sleep(1)
    subprocess.Popen([sys.executable, "simon/simon_says.py"])

def launch_snake():
    send_mode_to_arduino("SNAKE")
    time.sleep(1)
    subprocess.Popen([sys.executable, "snake/snake_with_ai.py"])

def launch_rps():
    subprocess.Popen([sys.executable, "Rock-Paper-Scissors/RPS_game.py"])


def main():
    root = tb.Window(themename="flatly")
    root.title("ROBOFRIEND")
    root.geometry("400x300")
    root.resizable(False, False)

    tb.Label(root, text="ROBOFRIEND", font=("Segoe UI", 22, "bold")).pack(pady=(30, 20))
    tb.Button(root, text="🎵 Simon Says", bootstyle=PRIMARY, width=25, command=launch_simon).pack(pady=10)
    tb.Button(root, text="🐍 Snake", bootstyle=SUCCESS, width=25, command=launch_snake).pack(pady=10)
    tb.Button(root, text="✊ Rock–Paper–Scissors", bootstyle=INFO, width=25, command=launch_rps).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
