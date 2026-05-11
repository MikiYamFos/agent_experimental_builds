import asyncio
from typing import Any, Dict
import json

import dotenv
import streamlit as st
from jaxn import JSONParserHandler, StreamingJSONParser
from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai.run import AgentRun
from pydantic_ai._agent_graph import UserPromptNode, ModelRequestNode, CallToolsNode

from doc_agent import DocumentationAgentConfig, create_agent
from models import RAGResponse
from tools import create_documentation_tools_cached

dotenv.load_dotenv()

GITHUB_DOCS_BASE_URL = "https://github.com/evidentlyai/docs/blob/main"


@st.cache_resource(show_spinner=False)
def get_agent() -> Agent:
    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()
    return create_agent(agent_config, tools)


class StreamlitRAGResponseHandler(JSONParserHandler):
    def __init__(self, message: Dict[str, Any], answer_placeholder):
        self.message = message
        self.answer_placeholder = answer_placeholder

    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        if path == "" and field_name == "answer":
            self.message["answer"] += chunk
            self.answer_placeholder.markdown(self.message["answer"] + "▌")

    def on_field_end(
        self,
        path: str,
        field_name: str,
        value: str,
        parsed_value: Any = None,
    ) -> None:
        if path != "":
            return

        if field_name == "answer_type":
            self.message["answer_type"] = value
        elif field_name == "confidence":
            self.message["confidence"] = (
                parsed_value if parsed_value is not None else value
            )
        elif field_name == "found_answer":
            self.message["found_answer"] = (
                parsed_value if parsed_value is not None else value
            )
        elif field_name == "confidence_explanation":
            self.message["confidence_explanation"] = value

    def on_array_item_end(
        self,
        path: str,
        field_name: str,
        item: Dict[str, Any] = None,
    ) -> None:
        if field_name == "followup_questions":
            self.message["followup_questions"].append(item)

        elif field_name == "references" and item:
            self.message["references"].append(item)


def render_references(message: Dict[str, Any]):
    references = message.get("references", [])

    if not references:
        return

    st.markdown("#### References")

    for reference in references:
        file_path = reference.get("file_path", "")
        explanation = reference.get("explanation", "")

        if not file_path:
            continue

        file_url = f"{GITHUB_DOCS_BASE_URL}/{file_path}"

        st.markdown(
            f"""
            <div style="
                border: 1px solid #333;
                border-radius: 0.5rem;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
                background-color: #111827;
            ">
                <a href="{file_url}" target="_blank" style="color: #93c5fd; font-weight: 600;">
                    {file_path}
                </a>
                <div style="color: #d1d5db; margin-top: 0.35rem;">
                    {explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_activity_panel(message: Dict[str, Any]):
    status = message.get("activity_status", "Done")
    actions = message.get("activity", [])

    with st.expander(status, expanded=True):
        if not actions:
            st.write("Thinking...")
            return

        for action in actions:
            if action["type"] == "search":
                st.write(f"🔎 Search: `{action['query']}`")
            elif action["type"] == "get_file":
                filename = action["filename"]
                file_url = f"{GITHUB_DOCS_BASE_URL}/{filename}"
                st.markdown(f"📄 File: [{filename}]({file_url})")


def render_metadata(message: Dict[str, Any]):
    answer_type = message.get("answer_type")
    confidence = message.get("confidence")
    found_answer = message.get("found_answer")

    if answer_type is None and confidence is None and found_answer is None:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"Type: {answer_type or 'Unknown'}")

    with col2:
        if isinstance(confidence, (int, float)):
            st.caption(f"Confidence: {confidence:.0%}")
        else:
            st.caption("Confidence: Unknown")

    with col3:
        if found_answer is True:
            st.caption("Found in docs: Yes")
        elif found_answer is False:
            st.caption("Found in docs: No")
        else:
            st.caption("Found in docs: Unknown")


def render_message(message: Dict[str, Any]):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_activity_panel(message)
            st.markdown(message.get("answer", ""))
            render_references(message)
            render_metadata(message)
        else:
            st.markdown(message["content"])


async def process_user_node(
    node: UserPromptNode,
    agent_run: AgentRun,
):
    return None


async def process_model_request_node(
    node: ModelRequestNode,
    agent_run: AgentRun,
    message: Dict[str, Any],
    answer_placeholder,
):
    args_so_far = ""
    parser = StreamingJSONParser(
        StreamlitRAGResponseHandler(
            message=message,
            answer_placeholder=answer_placeholder,
        )
    )

    async with node.stream(agent_run.ctx) as stream:
        async for response in stream.stream_responses():
            for part in response.parts:
                if part.part_kind != "tool-call":
                    continue

                if part.tool_name != "final_result":
                    continue

                args_new = part.args
                args_new_chunk = args_new[len(args_so_far) :]
                args_so_far = args_new

                parser.parse_incremental(args_new_chunk)


def parse_tool_args(args):
    if isinstance(args, dict):
        return args

    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}

    return {}


async def process_call_tools_node(
    node: CallToolsNode,
    agent_run: AgentRun,
    agent_name: str,
    message: Dict[str, Any],
    activity_placeholder,
):
    async with node.stream(agent_run.ctx) as events:
        async for event in events:
            if not isinstance(event, FunctionToolCallEvent):
                continue

            tool_name = event.part.tool_name
            args = parse_tool_args(event.part.args)

            if tool_name == "search":
                message["activity"].append(
                    {
                        "type": "search",
                        "query": args.get("query"),
                    }
                )

            elif tool_name == "get_file":
                message["activity"].append(
                    {
                        "type": "get_file",
                        "filename": args.get("filename"),
                    }
                )

            with activity_placeholder.container():
                render_activity_panel(message)


async def run_agent_for_ui(
    agent: Agent,
    user_prompt: str,
    message_history,
    message: Dict[str, Any],
    activity_placeholder,
    answer_placeholder,
):
    async with agent.iter(
        user_prompt,
        message_history=message_history,
        output_type=RAGResponse,
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_user_prompt_node(node):
                await process_user_node(node, agent_run)

            elif Agent.is_model_request_node(node):
                await process_model_request_node(
                    node=node,
                    agent_run=agent_run,
                    message=message,
                    answer_placeholder=answer_placeholder,
                )

            elif Agent.is_call_tools_node(node):
                await process_call_tools_node(
                    node=node,
                    agent_run=agent_run,
                    agent_name=agent.name,
                    message=message,
                    activity_placeholder=activity_placeholder,
                )

        message["activity_status"] = "Done"
        message["result"] = agent_run.result
        message["message_history"].extend(agent_run.result.new_messages())

        if agent_run.result.output:
            output = agent_run.result.output
            message["answer"] = output.answer
            message["found_answer"] = output.found_answer
            message["confidence"] = output.confidence
            message["confidence_explanation"] = output.confidence_explanation
            message["answer_type"] = output.answer_type
            message["followup_questions"] = output.followup_questions
            message["references"] = [
                reference.model_dump()
                for reference in output.references
            ]
        answer_placeholder.markdown(message["answer"])

        with activity_placeholder.container():
            render_activity_panel(message)


def submit_prompt(prompt: str):
    st.session_state.pending_prompt = prompt
    st.session_state.latest_followups = []


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "latest_followups" not in st.session_state:
        st.session_state.latest_followups = []


def main():
    st.set_page_config(
        page_title="Documentation Agent",
        page_icon="📚",
    )

    initialize_session_state()

    st.title("Documentation Agent")

    with st.sidebar:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.message_history = []
            st.session_state.pending_prompt = None
            st.session_state.latest_followups = []
            st.rerun()

    with st.spinner("Loading index..."):
        agent = get_agent()

    for message in st.session_state.messages:
        render_message(message)

    if st.session_state.latest_followups:
        st.write("Suggested follow-up questions:")
        columns = st.columns(len(st.session_state.latest_followups))

        for column, question in zip(columns, st.session_state.latest_followups):
            with column:
                if st.button(question, key=f"followup-{question}"):
                    submit_prompt(question)
                    st.rerun()

    typed_prompt = st.chat_input("Ask a question about Evidently docs")

    if typed_prompt:
        submit_prompt(typed_prompt)
        st.rerun()

    if not st.session_state.pending_prompt:
        return

    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    st.session_state.latest_followups = []

    user_message = {
        "role": "user",
        "content": user_prompt,
    }
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_prompt)

    assistant_message = {
        "role": "assistant",
        "answer": "",
        "activity": [],
        "activity_status": "Thinking...",
        "answer_type": None,
        "confidence": None,
        "found_answer": None,
        "confidence_explanation": None,
        "followup_questions": [],
        "references": [],
        "message_history": st.session_state.message_history,
    }
    st.session_state.messages.append(assistant_message)

    with st.chat_message("assistant"):
        activity_placeholder = st.empty()
        answer_placeholder = st.empty()

        with activity_placeholder.container():
            render_activity_panel(assistant_message)

        asyncio.run(
            run_agent_for_ui(
                agent=agent,
                user_prompt=user_prompt,
                message_history=st.session_state.message_history,
                message=assistant_message,
                activity_placeholder=activity_placeholder,
                answer_placeholder=answer_placeholder,
            )
        )

        render_references(assistant_message)
        render_metadata(assistant_message)

    st.session_state.latest_followups = assistant_message["followup_questions"]

    st.rerun()


if __name__ == "__main__":
    main()
