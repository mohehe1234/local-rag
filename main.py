import sys
import json
import argparse
import textwrap
from pathlib import Path
from typing import List
from datetime import datetime
from pydantic import BaseModel, RootModel, Field, ValidationError
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


def main():
    args = parse_args()
    cases = read_input(args)
    output_dir = prepare_output_dir(args)
    llm, model = setup_llm(args)

    if None in [cases, output_dir, llm]:
        sys.exit(1)

    log = {"model": model, "rag": args.rag}

    builder = WithRAG if args.rag else WithoutRAG
    runner = builder(llm, args.rek, log)

    with (output_dir / "log.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    cases_dir = output_dir / "cases"
    cases_dir.mkdir()

    for case in cases:
        print("Processing case", case["Case_number"])
        case["Stage"] = determine_stage(case)
        query = (
            textwrap.dedent("""
            You must complete Tasks 1\u20136 based on the image findings provided below.

            Task 1: Diagnose the local invasion factors of pancreatic cancer (CH, DU, S, RP, PV, A, PL, OO), and respond in the format: e.g., 'CH0, DU1, S1, RP1, PV0, A0, PL0, OO1'.

            Task 2: Based on the answer to Task 1, determine the classification for local invasion (T classification) of pancreatic cancer (T0, Tis, T1a, T1b, T1c, T2, T3, T4). If both size-based (e.g., T1, T2) and extension-based (e.g., T3, T4) criteria are met, assign the higher T category reflecting greater invasion.

            Task 3: Determine the N classification (N0, N1a, N1b) of pancreatic cancer. First, list the stations defined as regional lymph nodes based on the tumor location (head, body, or tail) according to the guideline. Then identify which of the patient\u2019s metastatic lymph node stations are regional, and exclude all others. Count only the metastatic lymph nodes\u2014not stations\u2014within the regional stations, and assign the N classification accordingly. Verify the total count of involved regional nodes.

            Task 4: Determine the M classification (M0 or M1) of pancreatic cancer, based on the presence or absence of distant metastases, such as to non-regional lymph nodes or distant organs. Do not confuse direct tumor invasion with distant metastasis.

            Task 5: Based on the results of Task 1 and Task 4, determine the resectability classification of pancreatic cancer as Resectable (R), Borderline Resectable (BR), or Unresectable (UR). Regional lymph node involvement does not qualify as distant or non-regional metastasis. If none of the criteria for BR or UR are met, classify as R.

            Task 6: Based on the determined T, N, and M categories, classify the overall stage according to the JPS staging system. Output one of: Stage 0, Stage IA, Stage IB, Stage IIA, Stage IIB, Stage III, or Stage IV.

            Image findings: {findings}
            """)
            .format(findings=case["Findings"])
            .strip()
        )
        before = datetime.now()
        responses = runner.run(query)
        after = datetime.now()
        duration = (after - before).total_seconds()
        structured_output_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a medical assistant. Below is the full diagnostic reasoning process for a pancreatic cancer patient across clinical tasks. Based only on this information, return a JSON object in the following format strictly adhering to each field's format.",
                ),
                ("placeholder", "{diagnostic_text}"),
                (
                    "system",
                    textwrap.dedent("""
                    Return only JSON:
                    {{
                      "CH": "CH0 or CH1",
                      "DU": "DU0 or DU1",
                      "S": "S0 or S1",
                      "RP": "RP0 or RP1",
                      "PV": "PV0 or PV1",
                      "A": "A0 or A1",
                      "PL": "PL0 or PL1",
                      "OO": "OO0 or OO1",
                      "T": "T0, Tis, T1a, T1b, T1c, T2, T3, or T4",
                      "N": "N0, N1a, or N1b",
                      "M": "M0 or M1",
                      "Resectability": "R, BR, or UR",
                      "Stage": "Stage 0, Stage IA, Stage IB, Stage IIA, Stage IIB, Stage III, or Stage IV"
                    }}
                    """).strip(),
                ),
            ]
        ) | llm.with_structured_output(DiagnosisOutput)
        structured_output = structured_output_chain.invoke(
            {"diagnostic_text": responses}
        )
        evaluations = {
            key: {
                "predicted": value,
                "ground truth": case[key],
                "matches": value == case[key],
            }
            for key, value in structured_output.model_dump().items()
        }
        case_path = cases_dir / f"{case['Case_number']}.json"
        case_path = case_path.resolve()
        with case_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "case_number": case["Case_number"],
                    "findings": case["Findings"],
                    "responses": [
                        {
                            "type": response.type,
                            "content": response.content,
                            "additional_kwargs": response.additional_kwargs,
                        }
                        for response in responses
                    ],
                    "evaluations": evaluations,
                    "duration": duration,
                },
                f,
                indent=2,
            )
        print(f"Output written to {case_path}.")


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="path to the input JSON file",
    )
    parser.add_argument(
        "output",
        help="directory to save the results",
    )
    parser.add_argument(
        "--rek",
        required=True,
        help="path to the reliable external knowledge (REK)",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="enable Retrieval-Augmented Generation (RAG)",
    )
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--model-openai", help="name of the OpenAI model to use")
    models.add_argument("--model-ollama", help="name of the Ollama model to use")
    parser.add_argument("--api-key", help="path to the file containing the API key")
    parser.add_argument("--case-min", type=int, help="minimum Case_number to include")
    parser.add_argument("--case-max", type=int, help="maximum Case_number to include")
    return parser.parse_args()


def read_input(args):
    """
    Reads and filters JSON-formatted case data
    based on the following command-line arguments.
    - input (positional): Path to the input JSON file.
    - --case-min: Minimum case number to include (inclusive).
    - --case-max: Maximum case number to include (inclusive).
    """
    try:
        with open(args.input, encoding="utf-8") as f:
            all_cases = json.load(f)
            ValidationModel(all_cases)
    except (ValidationError, json.JSONDecodeError) as e:
        print("Validation error: JSON does not match the expected format.", file=sys.stderr)
        print(e, file=sys.stderr)
        return None
    except OSError as err:
        print("Error: Cannot read input.", file=sys.stderr)
        print(err, file=sys.stderr)
        return None
    return [
        case
        for case in all_cases
        if (args.case_min is None or args.case_min <= case["Case_number"])
        and (args.case_max is None or case["Case_number"] <= args.case_max)
    ]


def prepare_output_dir(args):
    """
    Ensures the output directory exists and is empty. mkdir's if missing.
    """
    path = Path(args.output)
    try:
        if path.exists():
            if any(path.iterdir()):
                print(
                    f"Error: Directory `{path}` already exists and is not empty.",
                    file=sys.stderr,
                )
                return None
        else:
            path.mkdir(parents=True)
        return path
    except OSError as err:
        print("Error: Cannot create output directory.", file=sys.stderr)
        print(err, file=sys.stderr)
        return None


def setup_llm(args):
    """
    Initializes and returns a language model instance
    based on the following command-line arguments.
    - --model-ollama: Name of the Ollama model to use.
    - --model-openai: Name of the OpenAI model to use.
    - --api-key: Path to a file containing the OpenAI API key.
    """
    if args.model_openai is not None:
        if args.api_key is None:
            print("Error: Specify --api-key to use the OpenAI model.", file=sys.stderr)
            return None, None
        try:
            with open(args.api_key, encoding="utf-8") as f:
                openai_api_key = f.read().strip()
        except OSError as err:
            print("Error: Cannot read API key.", file=sys.stderr)
            print(err, file=sys.stderr)
            return None, None
        model = args.model_openai
        llm = ChatOpenAI(
            model=model,
            temperature=0.2,
            api_key=openai_api_key,
        )
    elif args.model_ollama is not None:
        model = args.model_ollama
        llm = ChatOllama(model=model, temperature=0.2)
    return llm, model


def determine_stage(case):
    if case["M"] == "M1":
        return "Stage IV"
    if case["T"] in {"T0", "Tis"} and case["N"] == "N0":
        return "Stage 0"
    if case["T"] in {"T1a", "T1b", "T1c"} and case["N"] == "N0":
        return "Stage IA"
    if case["T"] == "T2" and case["N"] == "N0":
        return "Stage IB"
    if case["T"] == "T3" and case["N"] == "N0":
        return "Stage IIA"
    if case["T"] in {
        "T0",
        "Tis",
        "T1a",
        "T1b",
        "T1c",
        "T2",
        "T3",
    } and case["N"] in {
        "N1a",
        "N1b",
    }:
        return "Stage IIB"
    if case["T"] == "T4" and case["M"] == "M0":
        return "Stage III"

class FindingsValidation(BaseModel):
    Case_number: int
    Findings: str
    CH: str = Field(pattern=r"^CH[01]$", description="Local invasion factor CH.")
    DU: str = Field(pattern=r"^DU[01]$", description="Local invasion factor DU.")
    S: str = Field(pattern=r"^S[01]$", description="Local invasion factor S.")
    RP: str = Field(pattern=r"^RP[01]$", description="Local invasion factor RP.")
    PV: str = Field(pattern=r"^PV[01]$", description="Local invasion factor PV.")
    A: str = Field(pattern=r"^A[01]$", description="Local invasion factor A.")
    PL: str = Field(pattern=r"^PL[01]$", description="Local invasion factor PL.")
    OO: str = Field(pattern=r"^OO[01]$", description="Local invasion factor OO.")
    T: str = Field(pattern=r"^T(0|is|1a|1b|1c|2|3|4)$", description="T classification.")
    N: str = Field(pattern=r"^N(0|1a|1b)$", description="N classification.")
    M: str = Field(pattern=r"^M[01]$", description="M classification.")
    Resectability: str = Field(
        pattern=r"^(R|BR|UR)$",
        description="Resectability classification.",
    )
class ValidationModel(RootModel):
    root: list[FindingsValidation]

class DiagnosisOutput(BaseModel):
    CH: str = Field(pattern=r"^CH[01]$", description="Local invasion factor CH.")
    DU: str = Field(pattern=r"^DU[01]$", description="Local invasion factor DU.")
    S: str = Field(pattern=r"^S[01]$", description="Local invasion factor S.")
    RP: str = Field(pattern=r"^RP[01]$", description="Local invasion factor RP.")
    PV: str = Field(pattern=r"^PV[01]$", description="Local invasion factor PV.")
    A: str = Field(pattern=r"^A[01]$", description="Local invasion factor A.")
    PL: str = Field(pattern=r"^PL[01]$", description="Local invasion factor PL.")
    OO: str = Field(pattern=r"^OO[01]$", description="Local invasion factor OO.")
    T: str = Field(pattern=r"^T(0|is|1a|1b|1c|2|3|4)$", description="T classification.")
    N: str = Field(pattern=r"^N(0|1a|1b)$", description="N classification.")
    M: str = Field(pattern=r"^M[01]$", description="M classification.")
    Resectability: str = Field(
        pattern=r"^(R|BR|UR)$",
        description="Resectability classification.",
    )
    Stage: str = Field(
        pattern=r"^Stage (0|IA|IB|IIA|IIB|III|IV)$",
        description="Stage classification.",
    )


class WithoutRAG:
    """
    A language model runner that performs diagnosis
    without using Retrieval-Augmented Generation (RAG).
    """

    def __init__(self, llm, rek_path, _):
        with open(rek_path, encoding="utf-8") as f:
            rek_text = f.read()
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    textwrap.dedent("""
                    You are a guideline-based decision assistant who answers each medical task strictly based on the provided guideline and patient information.
                    If the current task depends on previous task outputs, you MUST carefully refer to them to ensure consistency.
                    Do not infer conclusions from vague, partial, or unstated findings. Do not assume equivalence or relatedness between anatomical structures unless explicitly stated in the guideline.
                    """).strip(),
                ),
                (
                    "human",
                    "Answer based strictly on the following guideline content:\n\n"
                    + rek_text,
                ),
                ("human", "{query}"),
            ]
        )
        self.llm = prompt_template | llm

    def run(self, query):
        response = self.llm.invoke(query)
        return [response]


class SubtaskList(BaseModel):
    subtasks: List[str]
    context: str


class WithRAG:
    """
    A language model runner that performs diagnosis
    using Retrieval-Augmented Generation (RAG).
    """

    def __init__(self, llm, rek_path, log):
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        docs = TextLoader(rek_path, encoding="utf-8").load()
        splitter = MarkdownTextSplitter(
            chunk_size=1500,
            chunk_overlap=500,
            keep_separator=True,
        )
        split_docs = splitter.split_documents(docs)
        log["chunks"] = []
        for i, doc in enumerate(split_docs, 1):
            doc.metadata["chunk_id"] = str(i)
            log["chunks"].append({"chunk_index": i, "content": doc.page_content})
        db = FAISS.from_documents(split_docs, embedding_model)
        self.retriever = db.as_retriever(search_kwargs={"k": 4})

        split_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a clinical assistant that extracts structured subtasks and shared context from a complex instruction.",
                ),
                (
                    "human",
                    textwrap.dedent("""
                    Split the input into:
                    - a list of subtasks (each as a string),
                    - and the shared context (as a string).

                    **Important**:
                    - Keep all section headers and labels.
                    - Do **not** rewrite or summarize the content.
                    - If the instruction is a single block, return it as one subtask.

                    Return the result in this JSON format:
                    {{
                      "subtasks": ["<Task 1: ...>", "<Task 2: ...>", ...],
                      "context": "<Patient information relevant to all tasks>"
                    }}

                    Instruction:
                    {input}
                    """).strip(),
                ),
            ]
        )
        self.split_chain = split_prompt | llm.with_structured_output(SubtaskList)
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    textwrap.dedent("""
                    You are a guideline-based decision assistant who answers each medical task strictly based on the provided guideline and patient information.
                    If the current task depends on previous task outputs, you MUST carefully refer to them to ensure consistency.
                    Do not infer conclusions from vague, partial, or unstated findings. Do not assume equivalence or relatedness between anatomical structures unless explicitly stated in the guideline.
                    """).strip(),
                ),
                ("placeholder", "{history}"),
                ("placeholder", "{task}"),
                (
                    "human",
                    textwrap.dedent("""
                    Patient information:
                    {context}
                    """).strip(),
                ),
                (
                    "human",
                    textwrap.dedent("""
                    Answer based strictly on the following guideline content:
                    {guideline}
                    """).strip(),
                ),
            ]
        )
        self.rag_chain = rag_prompt | llm
        rerank_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert assistant that selects the most relevant guideline chunks for a medical task.",
                ),
                (
                    "human",
                    textwrap.dedent("""
                    Task:
                    {task}

                    Below are candidate guideline chunks, each with a [chunk_id:].
                    Select and return ONLY the 1\u20132 most relevant chunk_ids.

                    {chunks}

                    Respond ONLY with a comma-separated list of chunk_ids (e.g., '5, 12'):
                    """).strip(),
                ),
            ]
        )
        self.rerank_chain = rerank_prompt | llm

    def run(self, query):
        response = self.split_chain.invoke(query)

        subtasks = response.subtasks
        context = response.context

        history = []

        for i, subtask in enumerate(subtasks, 1):
            task_key = f"Step_{i}"
            relevant_docs = self.retriever.invoke(subtask)

            chunk_map = {}
            chunk_descriptions = []
            for idx, doc in enumerate(relevant_docs):
                chunk_id = str(doc.metadata.get("chunk_id", f"N/A_{idx}"))
                chunk_map[chunk_id] = doc
                chunk_descriptions.append(
                    f"[chunk_id: {chunk_id}]\n{doc.page_content.strip()}"
                )

            chunks_formatted = "\n\n".join(chunk_descriptions)
            rerank_result = self.rerank_chain.invoke(
                {"task": subtask, "chunks": chunks_formatted}
            )
            selected_ids = [
                cid.strip()
                for cid in rerank_result.content.split(",")
                if cid.strip() in chunk_map
            ]
            selected_docs = [chunk_map[cid] for cid in selected_ids]

            guideline_text = "\n\n".join(doc.page_content for doc in selected_docs)
            subtask = HumanMessage(content=subtask)
            subtask.additional_kwargs["task_key"] = task_key
            result = self.rag_chain.invoke(
                {
                    "task": [subtask],
                    "context": context,
                    "history": history,
                    "guideline": guideline_text,
                }
            )
            result.additional_kwargs["chunks"] = selected_ids
            history += [subtask, result]
        return history


if __name__ == "__main__":
    main()
