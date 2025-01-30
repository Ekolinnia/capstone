from recognize_speech_commands import process_transcript
from navigation import load_places, save_place, navigate_to_place

def process_command():
    """Processes user commands and starts navigation."""
    print("Hey Walking...")

    transcription = process_transcript()
    if transcription:
        print(f"User: {transcription}")

        if "take me to" in transcription or "take me" in transcription:
            place_label = transcription.replace("take me to", "").replace("take me", "").strip()
            places_df = load_places()

            if place_label in places_df["places"].values:
                destination = places_df[places_df["places"] == place_label].iloc[0]
                navigate_to_place(place_label, destination["coordinatesx"], destination["coordinatesy"])
            else:
                print(f"What is the address of {place_label}?")
                address = process_transcript()
                if address:
                    save_place(place_label, address)

if __name__ == "__main__":
    while True:
        process_command()
