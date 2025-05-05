from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


async def read_stream_completion(response):
    collected_reasoning_content = []
    collected_messages = []
    tool_calls = []  # To accumulate tool call data

    role = None
    finish_reason = None
    chunk = None
    usage = None
    thinking = False

    # If chunk.choices is empty, print usage
    async for chunk in response:
        if not chunk.choices:
            usage = chunk.usage
        else:
            choice = chunk.choices[0]
            if choice.finish_reason != None:
                finish_reason = choice.finish_reason

            delta = choice.delta
            if delta.role != None:
                role = delta.role

            if (
                hasattr(delta, "reasoning_content")
                and delta.reasoning_content is not None
            ):
                if not thinking:
                    thinking = True
                    print("\n<thinking>\n")
                collected_reasoning_content.append(delta.reasoning_content)
                print(delta.reasoning_content, end="", flush=True)
                # Check if tool_calls are present in the delta
            elif hasattr(delta, "tool_calls") and delta.tool_calls:
                # tool_calls is expected to be a list of tool call chunks
                for tool_call in delta.tool_calls:
                    index = tool_call.index
                    if index == len(tool_calls):
                        tool_calls.append(
                            ChatCompletionMessageToolCall(
                                id=tool_call.id,
                                type=tool_call.type,
                                function=Function(
                                    name=tool_call.function.name,
                                    arguments=tool_call.function.arguments,
                                ),
                            )
                        )
                    else:
                        if tool_call.function.arguments != None:
                            tool_calls[
                                index
                            ].function.arguments += tool_call.function.arguments
            else:
                chunk_message = delta.content or ""
                collected_messages.append(chunk_message)
    if thinking:
        print("\n</thinking>\n")
    content = "".join(collected_messages)
    reasoning_content = (
        "".join(collected_reasoning_content) if collected_reasoning_content else None
    )

    completion = ChatCompletion(
        id=chunk.id,
        created=chunk.created,
        model=chunk.model,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    tool_calls=tool_calls if tool_calls else None,
                    role=role,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )
    # print(f"\ncompletion: {completion}")
    # print(f"\nreasoning_content: {reasoning_content}")
    return completion, reasoning_content
