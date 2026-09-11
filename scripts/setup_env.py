import os

key = None
env_paths = [r"D:\magicpin\.env", r"D:\razorpay\.env", r"D:\AI-Traige-Engine\.env"]
for p in env_paths:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        key = val
                        break
    if key:
        break

if key:
    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"OPENAI_API_KEY={key}\n")
        f.write("OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/\n")
        f.write("OPENAI_MODEL=gemini-2.5-flash\n")
    print("Successfully created .env with active API key and endpoint!")
else:
    print("Could not find key.")
