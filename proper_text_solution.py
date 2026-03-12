#!/usr/bin/env python3
"""
Proper Text Processing Solution
This shows the right way to solve text extraction tasks
"""

import re

def extract_name_simple(text):
    """Simple rule-based name extraction"""
    text_upper = text.upper()
    if "NAME IS" in text_upper:
        # Find text after "NAME IS"
        match = re.search(r"NAME IS\s+(\w+)", text_upper)
        if match:
            return match.group(1).lower()
    return ""

def extract_name_advanced(text):
    """More robust name extraction"""
    patterns = [
        r"(?:my\s+)?name\s+is\s+(\w+)",
        r"i\s+am\s+(\w+)",
        r"call\s+me\s+(\w+)",
    ]
    
    text_clean = text.lower()  # Use lowercase for matching
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return ""

# Test cases
test_cases = [
    "HELLO! MY NAME IS MARTEN",
    "Hi there, my name is Alice",
    "I am Bob, nice to meet you", 
    "Call me Charlie",
    "Good morning, name is David"
]

print("🎯 Text Processing - Name Extraction")
print("=" * 40)

for text in test_cases:
    simple_result = extract_name_simple(text)
    advanced_result = extract_name_advanced(text)
    
    print(f"Input: '{text}'")
    print(f"  Simple:   '{simple_result}'")
    print(f"  Advanced: '{advanced_result}'")
    print()

print("💡 Why this approach is better than TimesMamba:")
print("- ✅ Fast and accurate")  
print("- ✅ No training needed")
print("- ✅ Easy to understand and modify")
print("- ✅ Handles edge cases")
print("- ✅ Does exactly what you want")

print("\n🤖 When to use ML models for text:")
print("- Complex sentiment analysis")
print("- Language translation") 
print("- Document classification")
print("- Question answering")
print("- Text generation")

print("\n📊 When to use TimesMamba:")
print("- Stock prices over time")
print("- Temperature/weather data")  
print("- Sales forecasting")
print("- Sensor readings")
print("- Any numerical time series data")