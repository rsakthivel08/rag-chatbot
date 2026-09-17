# test_sarvam.py — run this standalone to isolate which API is failing
import os
from dotenv import load_dotenv
load_dotenv()

from sarvam_client import translate_to_english, translate_from_english, text_to_speech

# Test 1: Translation to English
print("=== TEST 1: Translate Tamil → English ===")
try:
    result = translate_to_english("வயிற்று வலி என்றால் என்ன?", "ta-IN")
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: Translation from English
print("\n=== TEST 2: Translate English → Tamil ===")
try:
    result = translate_from_english("Stomach pain in children can have many causes.", "ta-IN")
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 3: TTS
print("\n=== TEST 3: TTS Tamil ===")
try:
    audio = text_to_speech("வணக்கம். நான் விஓலா.", "ta-IN")
    with open("test_output.wav", "wb") as f:
        f.write(audio)
    print(f"SUCCESS: {len(audio)} bytes written to test_output.wav")
except Exception as e:
    print(f"FAILED: {e}")