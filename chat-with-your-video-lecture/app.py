from transcriber import extract_audio, transcribe
from chunker import chunk_transcript_with_timestamps
from vector_stoer import store_embedding, load_vector_store
from qa_chain import build_qa_chain
from utils import summarize_sources

import os

video_path = "sample_data/sample_lecture.mp4"
audio_path = extract_audio(video_path)
segments = transcribe(audio_path)

chunks = chunk_transcript_with_timestamps(segments)

db = store_embedding(chunks)

qa_chain = build_qa_chain(db)

query = "Summarize the key points from the first hour"
response = qa_chain({"query": query})

print("\n📌 Answer:\n", response["result"])
print("\n🔗 Source Segments:\n", summarize_sources(response["source_documents"]))
