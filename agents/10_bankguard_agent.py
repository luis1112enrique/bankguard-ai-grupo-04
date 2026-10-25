"""Terminal entry point for the BankGuard AI backend demo."""

from __future__ import annotations

import argparse
from datetime import date

from bankguard.compliance import extract_final_text
from bankguard.config import load_settings
from bankguard.orchestrator import (
    build_bankguard_runtime,
    compact_conversation_history,
    run_analysis,
)
from bankguard.transaction_tools import BankingDataRepository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BankGuard AI: análisis transaccional y cumplimiento PLD/FT"
    )
    parser.add_argument(
        "--client", help="Ejecuta un análisis estructurado, p. ej. C003"
    )
    parser.add_argument("--start", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--question", help="Pregunta normativa opcional")
    parser.add_argument("--no-compliance", action="store_true")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--no-alert", action="store_true")
    return parser


def _one_shot(args: argparse.Namespace) -> None:
    if args.start is None or args.end is None:
        raise SystemExit("--client requiere también --start y --end")
    result = run_analysis(
        args.client,
        args.start,
        args.end,
        compliance_question=args.question,
        include_compliance=not args.no_compliance,
        include_charts=not args.no_charts,
        generate_alert=not args.no_alert,
    )
    print(result.to_json(indent=2))


def _interactive() -> None:
    settings = load_settings()
    repository = BankingDataRepository.from_settings(settings)

    print("BankGuard AI — demo bancario multiagente")
    print("Datos ficticios. Los scores no son umbrales regulatorios.")
    print(
        "Clientes:",
        ", ".join(repository.client_ids),
        "| fecha de referencia:",
        repository.reference_date.isoformat(),
    )
    print("Escribe una consulta en español, o 'salir' para terminar.")
    conversation: list = []

    while True:
        try:
            question = input("\nPREGUNTA: ").strip()
        except EOFError:
            print("\nHasta luego.")
            return
        if not question:
            continue
        if question.casefold() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            return

        try:
            # Isolate run_id, traces and delegated evidence per top-level turn.
            runtime = build_bankguard_runtime(settings, repository=repository)
            result = runtime.banking_orchestrator.invoke(
                {
                    "messages": [
                        *conversation,
                        {"role": "user", "content": question},
                    ]
                }
            )
            answer = extract_final_text(result)
            print("\nRESPUESTA:")
            print(answer or "El agente no devolvió texto.")
            conversation = compact_conversation_history(
                result.get("messages", []),
                max_turns=4,
            )
        except Exception as exc:
            print(
                "\nERROR: No fue posible completar la consulta "
                f"({type(exc).__name__}). Verifica Ollama, el modelo configurado "
                "y, para preguntas normativas, Pinecone."
            )


def main() -> None:
    args = _parser().parse_args()
    if args.client:
        _one_shot(args)
    else:
        _interactive()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nHasta luego.")
    except Exception as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from None
