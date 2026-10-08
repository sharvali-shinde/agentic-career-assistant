import gradio as gr

from app.agent import CareerAgent


agent = CareerAgent()

gr.ChatInterface(
    agent.chat,
).launch()