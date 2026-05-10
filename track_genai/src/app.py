"""
track_genai/src/app.py

Gradio 6 UI for the Document Q&A assistant.

Usage (from project root):
    python track_genai/src/app.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from gemini_client import ask, ask_with_metadata, load_index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_document_list() -> str:
    try:
        docs = load_index()
        if not docs:
            return "No documents ingested yet. Run ingest.py first."
        lines = []
        for d in docs:
            size_kb = d["size_bytes"] / 1024
            lines.append(f"📄 **{d['filename']}** ({size_kb:.0f} KB)")
        return "\n".join(lines)
    except FileNotFoundError:
        return "No documents ingested yet. Run ingest.py first."


def format_history_for_client(gradio_history: list) -> list[dict]:
    """Convert Gradio 6 message dicts to gemini_client history format."""
    result = []
    for msg in gradio_history:
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        # ensure we always pass a plain string
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        result.append({"role": role, "text": str(content)})
    return result


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------


def chat(message: str, history: list, show_tokens: bool):
    """
    Gradio 6 chat handler.
    history is a list of {"role": "user"|"assistant", "content": str} dicts.
    Returns the updated history list.
    """
    if not message.strip():
        return history, ""

    client_history = format_history_for_client(history)

    try:
        if show_tokens:
            result = ask_with_metadata(message, history=client_history)
            answer = result["answer"]
            cost = (
                result["input_tokens"] / 1_000_000 * 0.15
                + result["output_tokens"] / 1_000_000 * 0.60
            )
            status = (
                f"Tokens — input: {result['input_tokens']:,}  "
                f"output: {result['output_tokens']:,}  "
                f"estimated cost: ${cost:.5f}"
            )
        else:
            answer = ask(message, history=client_history)
            status = ""

    except FileNotFoundError as e:
        answer = f"⚠️ {e}"
        status = ""
    except Exception as e:
        answer = f"⚠️ An error occurred: {e}"
        status = ""

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Document Q&A") as demo:
        gr.Markdown("# 📚 Document Q&A Assistant")
        gr.Markdown(
            "Ask questions about your ingested documents. "
            "Answers are grounded in the documents with page citations."
        )

        with gr.Row():
            # Left column — document info + settings
            with gr.Column(scale=1):
                gr.Markdown("### Ingested documents")
                doc_list = gr.Markdown(get_document_list())
                refresh_btn = gr.Button("↻ Refresh", size="sm", variant="secondary")

                gr.Markdown("### Settings")
                show_tokens = gr.Checkbox(label="Show token usage & cost", value=True)

                gr.Markdown("---")
                gr.Markdown(
                    "**Tips**\n"
                    "- Ask for specific sections or concepts\n"
                    "- Follow up with *'can you expand on that?'*\n"
                    "- Try *'summarise the key contributions'*"
                )

            # Right column — chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        label="",
                        scale=5,
                        lines=2,
                        max_lines=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                status_bar = gr.Markdown("")
                clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")

        # --------------- Event wiring ---------------

        send_btn.click(
            fn=chat,
            inputs=[msg_box, chatbot, show_tokens],
            outputs=[chatbot, status_bar],
        ).then(fn=lambda: "", outputs=msg_box)

        msg_box.submit(
            fn=chat,
            inputs=[msg_box, chatbot, show_tokens],
            outputs=[chatbot, status_bar],
        ).then(fn=lambda: "", outputs=msg_box)

        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, status_bar])
        refresh_btn.click(fn=get_document_list, outputs=doc_list)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[app] Starting Document Q&A assistant...")
    print("[app] Open http://localhost:7860 in your browser\n")
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
