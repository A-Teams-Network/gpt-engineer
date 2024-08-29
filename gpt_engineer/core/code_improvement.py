from gpt_engineer.core.documentation_loader import retrieve_docs, format_docs
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate


class AnswerSchema(BaseModel):

    code: str = Field(
        description="Suggested code improvement based on the provided documentation."
    )

    justification: str = Field(
        description="Explanation of the code improvement based on the provided documentation."
    )


def improve_code(code: str, filename: str) -> str:
    docs = retrieve_docs(code)

    if len(docs) > 0:
        chat = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
        )
        docs = retrieve_docs(code)
        documentation = format_docs(docs)
        template = PromptTemplate.from_template(
            """Your task is to resolve compatibility issues in the code snippet below. You receive a documentation and a code snippet. Your task is to analyze the code and improve it based on the provided documentation. If no improvement is needed, you can leave the code as is. Make as little changes as possible. If no changes are needed, you can leave the code as is. Ideally, you should not change the code at all. If you need to make changes, make sure they are minimal.
Never edit JSON or YAML files.
Do not generate any code comments.


# Documentation:
{documentation}
"""
        ).format(documentation=documentation)

        user_message = PromptTemplate.from_template(
            """# {filename}\n```{code}```"""
        ).format(filename=filename, code=code)

        messages = [
            SystemMessage(content=template),
            HumanMessage(content=user_message),
        ]

        structured = chat.with_structured_output(AnswerSchema)

        res = structured.invoke(messages)

        return res.code

    else:
        return code
