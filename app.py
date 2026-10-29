"""Entry point for Hugging Face Spaces (Gradio SDK)."""

from agent_judge.demo_app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
