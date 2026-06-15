import os

log_path = r"C:\Users\USER\.gemini\antigravity\brain\3373bcad-6c06-46da-b8f8-ff3a1f57bba4\.system_generated\tasks\task-4090.log"
if os.path.exists(log_path):
    print("Log size:", os.path.getsize(log_path))
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print("Total lines:", len(lines))
    print("First 20 lines:")
    for line in lines[:20]:
        print(line.strip())
    print("\nLast 20 lines:")
    for line in lines[-20:]:
        print(line.strip())
else:
    print("Log not found")
