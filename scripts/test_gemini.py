import os
from openai import OpenAI

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

print("GEMINI_API_KEY found:", bool(key))

client = OpenAI(
    api_key=key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Test with standard flash model
try:
    res = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Reply with 'GEMINI_ONLINE'"}],
        temperature=0.0
    )
    print("Gemini response:", res.choices[0].message.content.strip())
except Exception as e:
    print("gemini-2.5-flash error:", e)
    # Try gemini-1.5-flash
    try:
        res = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": "Reply with 'GEMINI_ONLINE'"}],
            temperature=0.0
        )
        print("Gemini 1.5 response:", res.choices[0].message.content.strip())
    except Exception as e2:
        print("gemini-1.5-flash error:", e2)
