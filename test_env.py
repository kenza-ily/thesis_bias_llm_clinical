import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("Python path:")
for path in sys.path:
    print(path)

print("\nEnvironment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

print("\nTrying to import some libraries:")
try:
    import numpy
    print("NumPy imported successfully")
except ImportError as e:
    print(f"Error importing NumPy: {e}")

try:
    import langchain
    print("Langchain imported successfully")
except ImportError as e:
    print(f"Error importing Langchain: {e}")

# Add more import tests for other libraries you need

print("\nCurrent working directory:", os.getcwd())
print("Contents of current directory:")
print(os.listdir('.'))