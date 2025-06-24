# input_picker.py
import argparse
import os
import tkinter as tk
from tkinter import filedialog


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file or directory")
    return parser.parse_args()


def pick_file_or_folder():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    choice = input("Pick [F]ile or [D]irectory? ").strip().lower()
    if choice == "f":
        return filedialog.askopenfilename(title="Select input PDF or image file")
    elif choice == "d":
        return filedialog.askdirectory(title="Select input directory")
    else:
        print("Invalid choice.")
        return None


def resolve_input(cfg):
    args = parse_args()
    # Prefer command-line, then config, then GUI picker
    input_path = args.input or cfg.get("input_pdf") or cfg.get("input_dir")
    if not input_path:
        input_path = pick_file_or_folder()
    if input_path:
        if os.path.isdir(input_path):
            cfg["input_dir"] = input_path
            cfg["batch_mode"] = True
        elif os.path.isfile(input_path):
            cfg["input_pdf"] = input_path
            cfg["batch_mode"] = False
        else:
            raise ValueError("Input must be a file or directory")
    else:
        raise ValueError("No input file or directory provided!")
    return cfg
