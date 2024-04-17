import spacy

# Завантаження моделей мов для SpaCy
nlp_ru = spacy.load("ru_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")
nlp_uk = spacy.load("uk_core_news_sm")


# Функція обробки тексту
def process_text(text):
    # Обробка тексту для російської мови
    doc_ru = nlp_ru(text)
    print("Russian tokens:", [token.text for token in doc_ru])

    # Обробка тексту для англійської мови
    doc_en = nlp_en(text)
    print("English tokens:", [token.text for token in doc_en])

    # Обробка тексту для української мови
    doc_uk = nlp_uk(text)
    print("Ukrainian tokens:", [token.text for token in doc_uk])


# Головна функція
def main():
    # Введення тексту користувачем
    text = input("Введіть текст: ")

    # Обробка введеного тексту
    process_text(text)


if __name__ == "__main__":
    main()
