import re
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

fpath = r"C:\Users\USER\.gemini\antigravity\brain\3a868d61-12f3-40ac-8b5e-21be6a10609d\.system_generated\steps\270\output.txt"

if os.path.exists(fpath):
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        
    # Find all regions with Cell outputs
    # Let's search for "Cell X output"
    pattern = r'uid=\d+_\d+\s+region\s+"Cell\s+(\d+)\s+output"(.*?)(?=uid=\d+_\d+\s+(region|heading|main|complementary|button|link)|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    print(f"Found {len(matches)} cell outputs:")
    for cell_num, cell_content, _ in matches:
        print(f"\n--- Output of Cell {cell_num} ---")
        # Extract StaticText values inside cell_content
        text_matches = re.findall(r'StaticText\s+"(.*?)"', cell_content)
        cleaned_text = " ".join([t.replace('\\"', '"').replace('\\n', '\n').strip() for t in text_matches if t.strip()])
        # Also look for any raw text in the content
        print(cleaned_text[:1500] + ("..." if len(cleaned_text) > 1500 else ""))
else:
    print("Step 270 output file not found")
