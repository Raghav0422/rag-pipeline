from langchain_text_splitters import TextSplitter,CharacterTextSplitter

def split_documents(documents, chunk_size=800, overlap=0):
    print("splitting documents into chunks")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i+1} of {len(chunks)}")
            print(f"Content Length: {len(chunk.page_content)} characters")
            print(f"Source: {chunk.metadata['source']}")
            print("Content: ")
            print(chunk.page_content)
            print("-"*100)

    
    if len(chunks) > 5:
        print(f"\n ..... and {len(chunks) - 5} more chunks")

    return chunks