"""`zeppelin-ask "вопрос"` — run one question through the graph."""

import argparse

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the zeppelin knowledge base")
    parser.add_argument("question", help="Question in Russian or English")
    parser.add_argument("--trace", action="store_true", help="Print graph steps")
    args = parser.parse_args()

    load_dotenv()
    from zeppelin_rag.graph import build_default_graph

    graph, _ = build_default_graph()
    state = graph.invoke({"question": args.question, "steps": []})

    print(state["answer"])
    print("\nИсточники:")
    for i, chunk in enumerate(state["documents"], start=1):
        print(f"  [{i}] {chunk.source} — {chunk.section}")
    if args.trace:
        print("\nШаги графа:")
        for step in state["steps"]:
            print(f"  {step}")


if __name__ == "__main__":
    main()
