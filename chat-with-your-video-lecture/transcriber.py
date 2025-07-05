import os
import whisper
import ffmpeg # uv add ffmpeg-python

def extract_audio(video_path: str, audio_path: str = "audio.mp3"):
    input_file = ffmpeg.input(video_path)  
    input_file.output(audio_path, vn=None, acodec='libmp3lame').run()  
    return audio_path

def transcribe(audio_path:str = "audio.mp3", device: str = "cuda"):
    model = whisper.load_model("base")

    res = model.transcribe(audio_path)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return res["text"] 

def call():
    path = extract_audio("./sample-data/videoplayback.mp4")
    text = transcribe()
    print(text)

if __name__ == "__main__":
    call()