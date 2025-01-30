import subprocess

# List of required Python packages
required_packages = [
    "sounddevice",   # Microphone input handling
    "vosk",          # Speech-to-text engine
    "queue",         # Thread-safe queue for handling transcriptions
    "pyttsx3",       # Text-to-speech synthesis
    "tensorflow",    # Deep learning framework for command recognition
    "pandas",        # Data handling
    "numpy",         # Numerical computing
    "pickle5",       # Object serialization
    "geopy",         # Geolocation and address conversion
    "navitpy",       # Navigation system integration
    "pyserial",      # Handling GPS module communication
    "pynmea2",       # Parsing GPS NMEA sentences
]

# Function to install packages and track errors
def install_packages():
    failed_packages = []

    for package in required_packages:
        print(f"Installing {package}...")
        try:
            # Try installing the package
            subprocess.run(["pip", "install", package], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ {package} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {package}: {e.stderr.decode()}")
            failed_packages.append(package)

    # Display summary of failed packages
    if failed_packages:
        print("\n🚨 The following packages failed to install:")
        for package in failed_packages:
            print(f"❌ {package}")
    else:
        print("\n✅ All packages installed successfully!")

if __name__ == "__main__":
    print("Starting installation of dependencies...\n")
    install_packages()
