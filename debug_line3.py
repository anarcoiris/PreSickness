import re

line3 = "19/7/24, 1:11 - <?>: 🌊"
pattern = r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}) - ([^:]+): (.*)$"
match = re.match(pattern, line3)
print(f"Line 3: '{line3}'")
if match:
    print(f"MATCH: {match.groups()}")
else:
    print("NO MATCH")
    # Debug why
    parts = line3.split(" - ")
    if len(parts) > 1:
        print(f"Timestamp part: '{parts[0]}'")
        rest = parts[1]
        print(f"Rest: '{rest}'")
        if ": " in rest:
            name, msg = rest.split(": ", 1)
            print(f"Name: '{name}'")
            print(f"Msg: '{msg}'")
        else:
            print("No ': ' in rest")
