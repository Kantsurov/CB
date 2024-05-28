import openai
from key import hashed_api_key

api_key = hashed_api_key
openai.api_key = api_key


def ask_openai(question):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=question,
            max_tokens=100
        )

        return response.choices[0].text.strip()
    except Exception as e:
        print("Error:", e)
        return None


def main():
    question = input("Доброго дня! Чим можу допомогти? ")
    answer = ask_openai(question)

    if answer:
        print("Відповідь:", answer)
    else:
        print("Вибачте, я не знаю відповіді(")


if __name__ == "__main__":
    main()
