#!/usr/bin/env python3
"""Gradio frontend for Codebase Intelligence Assistant."""
import os
import sys
import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_EXAMPLES = [
    ["Explain what this Kotlin code does", "fun fetchUser(id: Int): Flow<User> = flow { emit(api.getUser(id)) }"],
    ["Find bugs in this Python code", "def divide(a, b): return a / b if b != 0: return None"],
    ["Write unit tests for this Jetpack Compose function", "@Composable fun Greeting(name: String) { Text(\"Hello $name\") }"],
    ["Refactor this code for better performance", "data = [x**2 for x in range(100000)]; result = sum(filter(lambda x: x > 50, data))"],
]


def chat_response(message, history, instruction_type, system_context):
    try:
        payload = {
            "instruction": instruction_type + " " + system_context,
            "input": message,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        resp = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()["response"]
        else:
            return f"Error: {resp.status_code} - {resp.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to backend. Make sure the FastAPI server is running.\nRun: uvicorn main:app --host 0.0.0.0 --port 8000"
    except Exception as e:
        return f"Error: {str(e)}"


def build_app():
    with gr.Blocks(title="Codebase Intelligence Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""# Codebase Intelligence Assistant
Built with Mistral 7B, fine-tuned on AMD ROCm. Specializes in code understanding, generation, and Android/mobile development.
""")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, show_copy_button=True)
                instruction_type = gr.Dropdown(
                    choices=[
                        "Explain what this code does",
                        "Find potential bugs and errors",
                        "Refactor and improve the code",
                        "Write unit tests",
                        "Answer this general coding question",
                        "Debug and fix this code",
                    ],
                    value="Explain what this code does",
                    label="Task Type",
                )
                with gr.Row():
                    msg = gr.Textbox(placeholder="Enter your code or question...", lines=4, scale=4)
                    submit = gr.Button("Send", scale=1)
                gr.Examples(
                    examples=DEFAULT_EXAMPLES,
                    inputs=[msg, instruction_type],
                    label="Try these examples",
                )
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                system_context = gr.Textbox(
                    value="You are an expert coding assistant.",
                    label="System Context",
                    lines=2,
                )
                max_tokens = gr.Slider(64, 4096, value=2048, step=64, label="Max New Tokens")
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                clear = gr.Button("Clear Chat")

        def respond(message, history, instruction_type, system_context):
            response = chat_response(message, history, instruction_type, system_context)
            history.append((message, response))
            return "", history

        submit.click(respond, inputs=[msg, chatbot, instruction_type, system_context], outputs=[msg, chatbot])
        msg.submit(respond, inputs=[msg, chatbot, instruction_type, system_context], outputs=[msg, chatbot])
        clear.click(lambda: (None, []), outputs=[msg, chatbot])

        gr.Markdown("""
### Tips
- Paste code directly into the input box
- Select a task type to guide the response
- Add domain context in "System Context" (e.g., "Focus on Jetpack Compose and Android development.")
""")
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)