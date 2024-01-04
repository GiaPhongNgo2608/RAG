# eval RAG
# eval Retrieval
# eval Respone
import json

import numpy as np
from langchain.evaluation import QAEvalChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from utils import json_loader
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group

LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate.from_template(
    """\
Create one or more statements from each sentence in the given answer.

question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements in json:
{{
    "statements": [
        "Albert Einstein was born in Germany.",
        "Albert Einstein was best known for his theory of relativity."
    ]
}}

question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alcohol
statements in json:
{{
    "statements": [
        "Cadmium Chloride is slightly soluble in alcohol."
    ]
}}

question: Were Hitler and Benito Mussolini of the same nationality?
answer: Sorry, I can't provide answer to that question.
statements in json:
{{
    "statements": []
}}

question:{question}
answer: {answer}
statements in json:"""  # noqa: E501
)
CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """\
Verify if the information in the given context is useful in answering the question.

question: What are the health benefits of green tea?
context: 
This article explores the rich history of tea cultivation in China, tracing its roots back to the ancient dynasties. It discusses how different regions have developed their unique tea varieties and brewing techniques. The article also delves into the cultural significance of tea in Chinese society and how it has become a symbol of hospitality and relaxation.
verification:
{{"reason":"The context, while informative about the history and cultural significance of tea in China, does not provide specific information about the health benefits of green tea. Thus, it is not useful for answering the question about health benefits.", "verdict":"No"}}

question: How does photosynthesis work in plants?
context:
Photosynthesis in plants is a complex process involving multiple steps. This paper details how chlorophyll within the chloroplasts absorbs sunlight, which then drives the chemical reaction converting carbon dioxide and water into glucose and oxygen. It explains the role of light and dark reactions and how ATP and NADPH are produced during these processes.
verification:
{{"reason":"This context is extremely relevant and useful for answering the question. It directly addresses the mechanisms of photosynthesis, explaining the key components and processes involved.", "verdict":"Yes"}}

question:{question}
context:
{context}
verification:"""  # noqa: E501
)

ground_truth = []

class Evaluate():
    def __init__(self,llm, dataset):
        self.dataset = dataset
        self.llm = llm
    def cores_batch(self):
        question, answer, contexts = {
            self.dataset["question"],
            self.dataset["answer"],
            self.dataset["context"]
        }
        prompts = []
        for q, a in zip(question,answer):
            human_prompt = LONG_FORM_ANSWER_PROMPT.format(question=q, answer=a)
            prompts.append(ChatPromptTemplate.from_template([human_prompt]))
        result = self.llm(prompts)
        prompts = []
        for context, output in zip(contexts, result):
            pass
    def context_precision(self):
        prompts = []
        questions = self.dataset["question"]
        contexts = self.dataset["contexts"]
        for question, context in zip(questions, contexts):
            human_prompts = [
                ChatPromptTemplate.from_template(
                    [CONTEXT_PRECISION.format(qestion=question,context=c)]
                )
                for c in context
            ]
            prompts.extend(human_prompts)
        responses: list[list[str]] = []
        results = self.llm()
        responses = [[i.text for i in r] for r in results]
        context_lens = [len(ctx) for ctx in contexts]
        context_lens.insert(0,0)
        context_lens = np.cumsum(context_lens)
        grouped_responses = [
            responses[start:end] for start, end in zip(context_lens[:-1], context_lens[1:])
        ]
        scores = []
        for response in grouped_responses:
            response = [
                json_loader.safe_load(item,self.llm) for item in sum(response, [])
            ]
            response = [
                int("yes" in resp.get("verdict", " ").lower())
                if resp.get("verdict")
                else np.nan
                for resp in response
            ]
            denominator = sum(response) + 1e-10
            numerator = sum(
                [
                    (sum(response[: i + 1]) / (i + 1)) * response[i]
                    for i in range(len(response))
                ]
            )
            scores.append(numerator / denominator)

        return scores


if __name__ == '__main__':

    questions = ["What did the president say about Justice Breyer?",
                 "What did the president say about Intel's CEO?",
                 "What did the president say about gun violence?",
                 ]
    ground_truths = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                     ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
                     ["The president asked Congress to pass proven measures to reduce gun violence."]]


# eval RAG
# eval Retrieval
# eval Respone
# question, content, answrer_llm, ground_truth
    contexts = []
    questions = []
    ground_truths = []
    answers = []
    with open('/home/rb025/Documents/PVP.txt') as file:
        text = file.read()
    sections = text.split('\n')
    for i,section in enumerate(sections):
        if i == 2:
            break
        elif i %2 == 0 and i<(len(sections)-1):
            questions.append(section.strip())
        elif i%2!=0:
            ground_truths.append([section.strip()])


    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use two sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever()
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',
        retriever=vectorstore.as_retriever()
    )
    df = pd.DataFrame(columns=["Query", "Ground Truth", "Context", "Answer","Evaluate_response","Evaluate_response"])

    for query, ground_truth in zip (questions, ground_truths):
        context = [docs.page_content for docs in retriever.get_relevant_documents(query)]
        answer = rag_chain.invoke(query)
        contexts.append(context)
        answers.append(answer)
        #eval = Evaluate(context=context, question=query,answer=answer['result'],ground_truth=ground_truth)
        # evaluate_retrieval = eval.evaluate_retrieval()
        # #evaluate_response = eval.evaluate_response()
        # df = df.append({"Query": query, "Ground Truth": ground_truth, "Context": context, "Answer": answer['result'], "Evaluate_retrieval":evaluate_retrieval},
        #                ignore_index=True)
        # print("Eval done")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    dataset = Dataset.from_dict(data)

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()




    # df.to_excel("evaluation_results.xlsx", index=False)
    # print("Results saved to evaluation_results.xlsx")


