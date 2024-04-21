import ollama
import chromadb
import requests
from bs4 import BeautifulSoup
from CB.RAG.search import embedmodel, collection
from utilities import readtext, getconfig
from mattsollamatools import chunk_text_by_sentences


def main():
    # Параметри для з'єднання з ChromaDB та ім'я колекції
    chroma = chromadb.HttpClient(host="localhost", port=8000)
    collectionname = "rag"

    # Видалення колекції, якщо вона вже існує
    if any(collection.name == collectionname for collection in chroma.list_collections()):
        chroma.delete_collection("rag")

    # Створення нової колекції
    collection = chroma.get_or_create_collection(name="rag", metadata={"hnsw:space": "cosine"})

    # Отримання конфігурації та моделі
    config = getconfig()
    embedmodel = config["embedmodel"]

    # Збереження документів в колекцію
    save_documents_to_database(collection, embedmodel)


def get_answer_from_database(query):
    embedmodel = getconfig()["embedmodel"]
    mainmodel = getconfig()["mainmodel"]
    queryembed = ollama.embeddings(model=embedmodel, prompt=query)['embedding']
    relevant_docs = collection.query(query_embeddings=[queryembed], n_results=5)["documents"][0]
    docs = "\n\n".join(relevant_docs)
    model_query = f"{query} - Answer that question using the following text as a resource: {docs}"

    stream = ollama.generate(model=mainmodel, prompt=model_query, stream=True)

    for chunk in stream:
        if chunk["response"]:
            print(chunk['response'], end='', flush=True)


def chat_with_ollama(collection, main_model):
    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break

        # Збереження запиту користувача в базу даних
        save_user_query_to_database(collection, query)

        # Отримання відповіді від моделі ollama
        response = ollama.chat(model=main_model, messages=[{"role": "user", "content": query}])

        # Збереження відповіді в базу даних
        save_ollama_response_to_database(collection, response)

        # Виведення відповіді
        print("AI: ", response["message"]["content"])


def save_user_query_to_database(collection, query):
    collection.add([{"role": "user", "content": query}])


def save_ollama_response_to_database(collection, response):
    collection.add([{"role": "AI", "content": response["message"]["content"]}])


def save_documents_to_database(collection, embedmodel):
    # Шлях до теки з документами
    documents_folder = 'documents'

    # Отримання списку файлів у текі
    files = os.listdir(documents_folder)

    for filename in files:
        # Формування повного шляху до файлу
        filepath = os.path.join(documents_folder, filename)

        # Читання тексту з файлу
        text = readtext(filepath)

        # Розбивка тексту на шматки
        chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=7, overlap=0)

        print(f"Processing {filename} with {len(chunks)} chunks")

        # Збереження кожного шматка разом з його вбудовуванням у базу даних
        for index, chunk in enumerate(chunks):
            embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
            collection.add([filename + str(index)], [embed], documents=[chunk], metadatas={"source": filename})


def read_and_save_webpage_data(url):
    # Отримання тексту з веб-сторінки
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()

    # Розділення тексту на шматки
    chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=7, overlap=0)

    # Збереження кожного шматка в базі даних ChromaDB
    for index, chunk in enumerate(chunks):
        embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
        collection.add([url + str(index)], [embed], documents=[chunk], metadatas={"source": url})

    print(f"Data from {url} has been saved to the ChromaDB collection.")


if __name__ == "__main__":
    main()
