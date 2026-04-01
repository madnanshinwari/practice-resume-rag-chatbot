import sys
from chat import chat

WELCOME = """
╔══════════════════════════════════════════════╗
║         Resume Chatbot  📄🤖                 ║
║  Ask anything about the resume.              ║
║  Type 'exit' or 'quit' to stop.              ║
╚══════════════════════════════════════════════╝
"""


def main():
    print(WELCOME)

    # Warn if resume hasn't been ingested yet
    try:
        from retriever import get_collection
        col = get_collection()
        print(f"[Ready] {col.count()} resume chunks loaded.\n")
    except RuntimeError as e:
        print(f"[Error] {e}")
        print("Run:  python ingest.py\n")
        sys.exit(1)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Bye!]")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Bot: Goodbye! 👋")
            break

        try:
            answer = chat(question)
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print(f"[Error] {e}\n")


if __name__ == "__main__":
    main()
