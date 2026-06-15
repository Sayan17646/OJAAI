import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

subagent_dir = r"C:\Users\USER\.gemini\antigravity\brain\3a868d61-12f3-40ac-8b5e-21be6a10609d"
steps_dir = os.path.join(subagent_dir, ".system_generated", "steps")

if os.path.exists(steps_dir):
    folders = [int(f) for f in os.listdir(steps_dir) if f.isdigit()]
    folders.sort()
    if folders:
        max_folder = max(folders)
        fpath = os.path.join(steps_dir, str(max_folder), "output.txt")
        if os.path.exists(fpath):
            print(f"Reading full content of Step {max_folder} output:")
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read().strip()
                # Print the lines of the output file
                lines = content.split('\n')
                print(f"Total lines: {len(lines)}")
                # Print first 50 lines and last 50 lines
                print("--- FIRST 50 LINES ---")
                for line in lines[:50]:
                    print(line)
                print("\n--- LAST 50 LINES ---")
                for line in lines[-50:]:
                    print(line)
else:
    print("Steps dir does not exist")
