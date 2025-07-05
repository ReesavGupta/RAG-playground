from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

def chunk_transcript_with_timestamps(result, chunk_size=1000, chunk_overlap=200):
    """
    Chunk transcript while preserving precise timestamp information for each chunk
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    full_text = result["text"]
    segments = result["segments"]
    
    # Create a mapping of character positions to timestamps
    char_to_timestamp = create_char_timestamp_mapping(full_text, segments)
    
    # Create chunks from full text
    text_chunks = text_splitter.split_text(full_text)
    
    # Create documents with precise timestamp metadata
    documents = []
    char_position = 0
    
    for i, chunk in enumerate(text_chunks):
        # Find precise start and end timestamps for this chunk
        chunk_start_pos = char_position
        chunk_end_pos = char_position + len(chunk) - 1
        
        start_time = find_timestamp_at_position(char_to_timestamp, chunk_start_pos)
        end_time = find_timestamp_at_position(char_to_timestamp, chunk_end_pos)
        
        doc = Document(
            page_content=chunk.strip(),
            metadata={
                "chunk_id": i,
                "start_time": start_time,
                "end_time": end_time,
                "duration": round(end_time - start_time, 2) if end_time and start_time else None,
                "char_start": chunk_start_pos,
                "char_end": chunk_end_pos
            }
        )
        documents.append(doc)
        
        # Move to next chunk position (accounting for overlap)
        char_position += len(chunk) - chunk_overlap
    
    return documents

def create_char_timestamp_mapping(full_text, segments):
    """
    Create a mapping of character positions to timestamps
    """
    char_to_timestamp = {}
    current_pos = 0
    
    for segment in segments:
        segment_text = segment["text"]
        segment_start = segment["start"]
        segment_end = segment["end"]
        
        # Find where this segment appears in the full text
        # Handle potential whitespace differences
        segment_clean = segment_text.strip()
        
        # Find the segment in the full text starting from current position
        search_start = max(0, current_pos - 50)  # Small buffer for overlap
        found_pos = full_text.find(segment_clean, search_start)
        
        if found_pos != -1:
            # Map each character in this segment to its timestamp
            segment_duration = segment_end - segment_start
            segment_length = len(segment_clean)
            
            for i, char in enumerate(segment_clean):
                char_pos = found_pos + i
                # Linear interpolation for timestamp within segment
                if segment_length > 0:
                    timestamp = segment_start + (i / segment_length) * segment_duration
                else:
                    timestamp = segment_start
                
                char_to_timestamp[char_pos] = timestamp
            
            current_pos = found_pos + len(segment_clean)
    
    return char_to_timestamp

def find_timestamp_at_position(char_to_timestamp, position):
    """
    Find the timestamp for a given character position
    """
    # First try exact match
    if position in char_to_timestamp:
        return char_to_timestamp[position]
    
    # If no exact match, find the closest position
    positions = sorted(char_to_timestamp.keys())
    
    if not positions:
        return 0.0
    
    # Find closest position
    closest_pos = min(positions, key=lambda x: abs(x - position))
    return char_to_timestamp[closest_pos]
