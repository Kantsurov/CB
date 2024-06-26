def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    # open file
    filename = "peter-pan.txt"
    paragraphs = parse_file(filename)

    embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)

    prompt = input("what do you want to know? -> ")
    # strongly recommended that all embeddings are generated by the same model (don't mix and match)
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
    # find most similar to each other
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                           + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("\n\n")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
