"""Run one isolated PaperQA query; executed with the PaperQA checkout's Python."""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    from paperqa.agents import ask
    from paperqa.settings import Settings

    settings = Settings.from_name("deepseek")
    settings.agent.index.paper_directory = args.papers
    settings.agent.index.index_directory = args.index
    settings.agent.index.name = "case"
    result = ask(args.question, settings)
    print(json.dumps({
        "answer": result.session.answer,
        "formatted_answer": result.session.formatted_answer,
        "status": str(result.status),
        "model": settings.llm,
        "summary_model": settings.summary_llm,
        "agent_model": settings.agent.agent_llm,
        "embedding": settings.embedding,
    }))


if __name__ == "__main__":
    main()
