import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Milvus
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    # 개인정보 보호법
    COLLECTION_NAME = 'medical_law_index'
    URI = os.environ["MILVUS_CLUSTER_ENDPOINT"]
    TOKEN = os.environ["MILVUS_TOKEN"]
    connection_args = { 'uri': URI, 'token': TOKEN }

    # Milvus Vector Store 초기화
    database = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args=connection_args
    )

    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    # 개인정보 보호법
    # dictionary = [
    #     "개인정보 보호법 위반을 나타내는 표현 -> 개인정보 보호법 위반",
    #     "개인정보 주체의 권리를 나타내는 표현 -> 개인정보 주체 권리",
    #     "개인정보 유출을 나타내는 표현 -> 개인정보 유출",
    #     "불법 수집을 나타내는 표현 -> 불법 수집",
    #     "정보 보안을 나타내는 표현 -> 정보 보안"
    # ]
    # 의료법
    dictionary = [
        "의료기록의 관리와 보호를 나타내는 표현 -> 의료기록 관리",
        "의료사고를 나타내는 표현 -> 의료사고",
        "의료진의 법적 책임을 나타내는 표현 -> 의료진 법적 책임",
        "환자의 권리를 나타내는 표현 -> 환자 권리",
        "의료정보의 보호를 나타내는 표현 -> 의료정보 보호"
    ]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.
        사전: {dictionary}

        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def get_rag_chain():
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
        input_variables=['input']
    )
    # 개인정보 보호법
    # system_prompt = (
    #     "당신은 개인정보 보호법 전문가입니다. 사용자의 개인정보 보호법에 관한 질문에 답변해 주세요. "
    #     "아래에 제공된 문서를 활용해서 답변해 주시고, "
    #     "답변을 알 수 없다면 모른다고 답변해 주세요. "
    #     "답변을 제공할 때는 '개인정보 보호법 (XX조)에 따르면' 이라고 시작하면서 답변해 주시고, "
    #     "사용자가 명쾌하게 이해할 수 있는 내용의 답변을 원합니다. "
    #     "답변의 내용에 '개인정보 주체의 권리', '개인정보 수집', '개인정보 유출', '정보 보안'에 관한 내용도 추가해서 답변해 주세요. "
    #     "ChatGPT 보다 나은 답변이 나온다면 당신은 두둑한 보너스를 받게 됩니다."
    #     "\n\n"
    #     "{context}"
    # )
    # 의료법
    system_prompt = (
        "당신은 의료법 전문가입니다. 사용자의 의료법에 관한 질문에 답변해 주세요. "
        "아래에 제공된 문서를 활용해서 답변해 주시고, "
        "답변을 알 수 없다면 모른다고 답변해 주세요. "
        "답변을 제공할 때는 '의료법 (XX조)에 따르면' 이라고 시작하면서 답변해 주시고, "
        "사용자가 명쾌하게 이해할 수 있는 내용의 답변을 원합니다. "
        "답변의 내용에는 '의료기관 개설 기준', '의료진 자격 요건', '의료기관의 광고 규제', '환자 권리 보호'에 관한 내용도 추가해서 답변해 주세요. "
        "ChatGPT보다 나은 답변이 나온다면 당신은 두둑한 보너스를 받게 됩니다."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain


def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    law_chain = {"input": dictionary_chain} | rag_chain
    ai_response = law_chain.stream(
        {
            "question": user_message
        },
        config=
        {
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response
