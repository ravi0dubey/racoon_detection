from pathlib import Path

input_source = r'C:/Users/Ravi0dubey/Videos/Bandicam'
video_files = [str(f) for f in Path(input_source).glob('**/*') if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')]
print(f"inside video input, video_files:{video_files}")