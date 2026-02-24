file_path = r"c:\Users\soyko\Documents\PreSickness\datos\paciente1_whatsapp.txt"
try:
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        for i in range(5):
            print(f"Line {i}: {lines[i]}")
            print(f"Hex: {lines[i].hex()}")
except Exception as e:
    print(e)
