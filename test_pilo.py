# test_pil.py

try:
    import PIL.Image
    print("PIL is installed correctly.")
except ImportError as e:
    print(f"Error: {e}")
