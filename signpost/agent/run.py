from __future__ import annotations

"""CLI entry point for F15 Supervisor-Researcher agent."""

import argparse
import json
import os

from signpost.agent.supervisor import AgentConfig, Supervisor
from signpost.agent.tools import ReadFileConfig, ReadFileTool, default_search_config, KnowledgeSearchTool
from signpost.config.context import resolve_project_path
from signpost.llm.client import OpenAICompatibleClient
from signpost.retrieval.signpost_variants import FULL, VALID_VARIANTS


def run_agent(
    *,
    namespace: str,
    question: str,
    dataset: str | None = None,
    embedding_provider: str = "ecnu",
    use_llm: bool = True,
    use_es: bool = False,
    max_subquestions: int = 3,
    read_top_k: int = 3,
    signpost_variant: str = FULL,
) -> dict:
    artifact_dataset = dataset or namespace
    search_tool = KnowledgeSearchTool(
        default_search_config(
            namespace,
            dataset=artifact_dataset,
            use_es=use_es,
            embedding_provider_name=embedding_provider,
            signpost_variant=signpost_variant,
        )
    )
    read_tool = ReadFileTool(ReadFileConfig(dataset=artifact_dataset))
    llm_timeout = int(os.environ.get("LLM_TIMEOUT", "600") or 600)
    llm = OpenAICompatibleClient(timeout=llm_timeout) if use_llm else None
    supervisor = Supervisor(
        AgentConfig(namespace=namespace, max_subquestions=max_subquestions, read_top_k=read_top_k, use_llm=use_llm),
        search_tool,
        read_tool,
        llm=llm,
    )
    return supervisor.run(question)


def main() -> int:
    parser = argparse.ArgumentParser(description="F15 Supervisor-Researcher Agent")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dataset", help="Processed dataset id for graph/chunks/documents. Defaults to --namespace.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "ecnu"],
        default="ecnu",
        help="Query embedding provider. Must match the ES index embedding provider when --use-es is set.",
    )
    parser.add_argument("--use-llm", dest="use_llm", action="store_true", help="Use ECNU/OpenAI-compatible chat for planning and synthesis.")
    parser.add_argument("--no-use-llm", dest="use_llm", action="store_false", help="Use deterministic decomposition and template synthesis for debugging.")
    parser.set_defaults(use_llm=True)
    parser.add_argument("--use-es", action="store_true", help="Use Elasticsearch retrieval instead of local artifacts.")
    parser.add_argument("--max-subquestions", type=int, default=3)
    parser.add_argument("--read-top-k", type=int, default=3)
    parser.add_argument("--signpost-variant", choices=sorted(VALID_VARIANTS), default=FULL)
    args = parser.parse_args()

    result = run_agent(
        namespace=args.namespace,
        question=args.question,
        dataset=args.dataset,
        embedding_provider=args.embedding_provider,
        use_llm=args.use_llm,
        use_es=args.use_es,
        max_subquestions=args.max_subquestions,
        read_top_k=args.read_top_k,
        signpost_variant=args.signpost_variant,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"output={output} trace_id={result['trace_id']} citations={len(result['citations'])}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
