def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def summarize_sources(sources):
    output = []
    for doc in sources:
        start = format_timestamp(doc.metadata['start'])
        end = format_timestamp(doc.metadata['end'])
        output.append(f"[{start} - {end}] {doc.page_content[:100]}...")
    return "\n".join(output)
