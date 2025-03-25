import os
import glob

def split_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks of roughly `chunk_size` words
    with `overlap` words carried over to the next chunk.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    input_directory = "extracted_texts"
    output_file = os.path.join(input_directory, "chunks.txt")
    
    # Get all .txt files in the directory
    input_files = glob.glob(os.path.join(input_directory, "*.txt"))
    
    all_chunks = []
    for file_path in input_files:
        # Skip the 'chunks.txt' if it exists
        if os.path.basename(file_path) == "chunks.txt":
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = split_text(text)
        all_chunks.extend(chunks)
        print(f"Processed '{os.path.basename(file_path)}' into {len(chunks)} chunks.")
    
    # Save all chunks to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk + "\n====\n")
            
    print(f"Created {len(all_chunks)} text chunks from {len(input_files)} files.")
