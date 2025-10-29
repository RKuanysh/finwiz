from rag_system import RAGSystem

PDF_PATH = "docs/nvidia_10K.pdf"

def main():
    rag = RAGSystem()
    rag.prepare(PDF_PATH)

    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.lower() in ("exit", "quit"):
            break
        answer = rag.answer_question(query)
        print(f"\n💡 Answer:\n{answer}")

if __name__ == "__main__":
    main()
