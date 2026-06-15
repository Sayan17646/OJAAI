import os

search_dir = r"C:\Users\USER\Desktop\OJAAI"
print("Searching for DevToolsActivePort in:", search_dir)
for root, dirs, files in os.walk(search_dir):
    for f in files:
        if "devtoolsactiveport" in f.lower():
            print(os.path.join(root, f))
