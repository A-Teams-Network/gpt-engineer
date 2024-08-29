from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import PythonCodeTextSplitter
from langchain.docstore.document import Document
import re


llm = ChatOpenAI(model="gpt-4o-mini")


# global vectorstore
vectorstore = None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_code_blocks(doc):
    regex = r"(\S+)\n\s*```[^\n]*\n(.+?)```"
    matches = re.finditer(regex, doc, re.DOTALL)

    code_blocks = []
    for match in matches:
        path = re.sub(r'[\:<>"|?*]', "", match.group(1))
        path = re.sub(r"^\[(.*)\]$", r"\1", path)
        path = re.sub(r"^`(.*)`$", r"\1", path)
        path = re.sub(r"[\]\:]$", "", path)
        content = match.group(2)
        stripped = content.strip()
        doc = Document(page_content=stripped, metadata={"source": "local"})

        code_blocks.append(doc)

    return code_blocks


def create_loader(url: str):
    if url.startswith("http"):
        return WebBaseLoader(url)
    else:
        raise ValueError("Unsupported URL scheme.")


def load_documents(url: str):
    global vectorstore

    if vectorstore is not None:
        return vectorstore

    loader = create_loader(url)
    docs = loader.load()
    doc = docs[0].page_content
    code_blocks = extract_code_blocks(doc)
    splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(code_blocks)

    return splits
    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())


def retrieve_docs(query: str) -> list[Document]:
    global vectorstore

    if vectorstore is None:
        raise ValueError("Vectorstore is not initialized.")

    docs = vectorstore.similarity_search(query, k=5)

    return docs
