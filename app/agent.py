"""Career chatbot agent orchestration."""
import json

from openai import OpenAI

from .config import NAME, LINKEDIN_URL, OPENAI_MODEL
from .retrieval import ProfileKnowledge
from .tools import TOOLS, TOOL_FUNCTIONS


class CareerAgent:

    def __init__(self):
        self.openai = OpenAI()
        self.name = NAME
        self.profile = ProfileKnowledge()

    def system_prompt(self):
        return f"""
You are acting as {self.name}. You are answering questions on
{self.name}'s website, particularly questions related to
{self.name}'s career, background, skills and experience.

Your responsibility is to represent {self.name} for interactions
on the website as faithfully as possible.

You are given a summary of {self.name}'s background and LinkedIn
profile which you can use to answer questions.

Be professional and engaging, as if talking to a potential client
or future employer who came across the website.

If you don't know the answer to any question, use your
record_unknown_question tool to record the question that you
couldn't answer, even if it's about something trivial or unrelated
to career.

Do not ask users any questions about their interests, hobbies, etc.

When asked about your background, be biased towards the work done
in the AI/Agentic domain.

If the user is sent 3 or more chats and is engaging in discussion,
try to steer them towards getting in touch via email; ask for their
email and record it using your record_user_details tool.

If the user refuses to provide an email id, ask them to shoot a
message on {self.name}'s LinkedIn.

{self.profile.get_context()}

With this context, please chat with the user, always staying in
character as {self.name}.
"""

    def handle_tool_call(self, tool_calls):
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"Tool called: {tool_name}", flush=True)

            tool = TOOL_FUNCTIONS.get(tool_name)

            if tool:
                result = tool(**arguments)
            else:
                result = {}

            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })

        return results

    def chat(self, message, history):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt(),
            }
        ] + history + [
            {
                "role": "user",
                "content": message,
            }
        ]

        done = False

        while not done:
            response = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
            )

            if response.choices[0].finish_reason == "tool_calls":
                assistant_message = response.choices[0].message
                tool_calls = assistant_message.tool_calls

                results = self.handle_tool_call(tool_calls)

                messages.append(assistant_message)
                messages.extend(results)

            else:
                done = True

        return response.choices[0].message.content