import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

subagent_dir = r"C:\Users\USER\.gemini\antigravity\brain\3a868d61-12f3-40ac-8b5e-21be6a10609d"
steps_dir = os.path.join(subagent_dir, ".system_generated", "steps")

if os.path.exists(steps_dir):
    # Sort folders numerically
    folders = []
    for item in os.listdir(steps_dir):
        if item.isdigit():
            folders.append(int(item))
    folders.sort()
    
    for f in folders:
        fpath = os.path.join(steps_dir, str(f), "output.txt")
        if os.path.exists(fpath):
            print(f"\n=================== Step {f} ===================")
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as file:
                print(file.read().strip()[:1000])
else:
    print("Steps dir does not exist")
