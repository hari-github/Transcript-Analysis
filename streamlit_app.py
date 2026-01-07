import streamlit as st
import pandas as pd
import os
import shutil
import random
import io
import json  # Fixed: Added missing import
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage

import ssl


# This bypasses SSL certificate verification for the entire session
if getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

# Additional bypass for common libraries like 'requests'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# --- CONSTANTS & PATHS ---
BASE_DIR = "transcript_data"
DISCOVERY_DIR = os.path.join(BASE_DIR, "discovery")
# Per requirement: remaining_dir is the base directory
REMAINING_DIR = BASE_DIR

# --- HELPER FUNCTIONS ---

def load_transcripts(folder_path: str) -> Dict[str, str]:
    transcripts = {}
    if not os.path.exists(folder_path):
        return transcripts

    # We only load files directly in the specified folder (not subfolders)
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and os.path.isfile(os.path.join(folder_path, f))]
    for file in files:
        transcript_id = os.path.splitext(file)[0]
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            transcripts[transcript_id] = f.read()
    return transcripts

def normalize_facts_output(facts):
    if facts is None:
        return {}
    if isinstance(facts, dict):
        return facts
    if isinstance(facts, list):
        normalized = {}
        for item in facts:
            if isinstance(item, dict) and "key" in item and "value" in item:
                normalized[item["key"]] = item["value"]
        return normalized
    return {}

def extract_facts(transcripts: dict[str, str], chain) -> dict[str, dict]:
    extracted = {}
    for transcript_id, text in transcripts.items():
        # chain is passed in to ensure scope is correct
        raw_facts = chain.invoke({"transcript": text})
        facts = normalize_facts_output(raw_facts)
        extracted[transcript_id] = facts
    return extracted

def build_fact_table(extracted_data: dict[str, dict]) -> pd.DataFrame:
    all_keys = set()
    for facts in extracted_data.values():
        all_keys.update(facts.keys())

    table = {}
    for key in sorted(all_keys):
        table[key] = {
            tid: extracted_data[tid].get(key)
            for tid in extracted_data
        }
    df = pd.DataFrame.from_dict(table, orient="index")
    df.index.name = "fact_key"
    return df

def force_string_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Uses the now-imported json library
    df.index = df.index.map(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
    return df

def generate_key_mapping(fact_keys: list[str], llm) -> dict:
    prompt = PromptTemplate(
        template="""You are a data normalization expert. Identify keys that represent the same concept...
        Fact keys: {keys}
        Return ONLY valid JSON.""",
        input_variables=["keys"]
    )
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"keys": fact_keys})

def first_non_null(values):
    for v in values:
        if v is None: continue
        if isinstance(v, float) and pd.isna(v): continue
        return v
    return None

def normalize_fact_dataframe(df: pd.DataFrame, key_mapping: dict) -> pd.DataFrame:
    df = df.copy()
    df["canonical_key"] = df.index.map(lambda k: key_mapping.get(k, k))
    normalized_df = df.groupby("canonical_key", dropna=False).agg(first_non_null)
    normalized_df.index.name = "fact_key"
    return normalized_df

def _flatten_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [str(v).strip() for v in value if v is not None and str(v).strip() != ""]
    return [str(value).strip()]

def build_fact_sample_view(df: pd.DataFrame, max_samples: int = 5) -> pd.DataFrame:
    rows = []
    for fact_key, row in df.iterrows():
        collected = []
        for cell in row:
            collected.extend(_flatten_value(cell))
        unique_values = list(dict.fromkeys([v for v in collected if v]))
        if unique_values:
            rows.append({"fact_key": fact_key, "sample_values": unique_values[:max_samples]})
    return pd.DataFrame(rows)

def extract_controlled_entities(transcripts_dir: str, final_entities: List[str], entity_chain) -> Dict[str, Dict]:
    extracted: Dict[str, Dict] = {}
    entity_list_str = "\n".join(f"- {e}" for e in final_entities)
    # We only want files in the base dir, not the discovery sub-folder
    txt_files = [f for f in Path(transcripts_dir).glob("*.txt") if f.is_file()]

    for file_path in txt_files:
        transcript_id = file_path.stem
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            print (text)
            time.sleep(3) # Reduced sleep for efficiency
            result = safe_entity_invoke(
            entity_chain,
            {
                "entity_list": entity_list_str,
                "transcript": text
            },
            transcript_id=transcript_id
            )

            print (result)
            print (entity_list_str)
            print ('-----------------------------------------------')
            extracted[transcript_id] = result if isinstance(result, dict) else {}
        except Exception as e:
            extracted[transcript_id] = {}
            print (e)
    return extracted

def build_entity_table(extracted_data: dict[str, dict], final_entities) -> pd.DataFrame:
    df = pd.DataFrame(extracted_data).reindex(final_entities)
    df.index.name = "entity"
    return df


def safe_parse_json(text: str) -> dict:
    """
    Attempts to extract and repair JSON from LLM output.
    Returns {} if parsing fails.
    """
    if not text:
        return {}

    # Remove markdown fences
    text = re.sub(r"^```json|```$", "", text.strip(), flags=re.IGNORECASE)

    # Extract JSON block if extra text exists
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except Exception:
        return {}


def safe_entity_invoke(
    entity_chain,
    inputs: dict,
    transcript_id: str,
    max_retries: int = 1
) -> dict:
    """
    Safely invokes an entity extraction chain.
    Never raises.
    Always returns a dict.
    """

    for attempt in range(max_retries + 1):
        try:
            result = entity_chain.invoke(inputs)

            # Case 1: JsonOutputParser → dict
            if isinstance(result, dict):
                return result

            # Case 2: AIMessage → parse content
            if isinstance(result, AIMessage):
                return safe_parse_json(result.content)

            # Case 3: raw string
            if isinstance(result, str):
                return safe_parse_json(result)

            # Unknown type
            return {}

        except Exception as e:
            if attempt == max_retries:
                print(
                    f"❌ Entity extraction failed for {transcript_id}: {e}"
                )
            continue

    return {}


# --- UI CONFIG ---
st.set_page_config(page_title="Call Center Analytics", layout="wide")

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'final_step' not in st.session_state:
    st.session_state.final_step = False

st.title("📞 Call Center Transcript Analytics")

# --- STEP 1: UPLOAD AND SPLIT ---
with st.sidebar:

    model_options = ["llama-3.3-70b-versatile", "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]


    # user_api_key = st.text_input("Enter the Groq API key", type="password")
    user_api_key = st.secrets["GROQ_API_KEY"]
    # Create the dropdown
    llm_model = st.selectbox("Choose a Model:", model_options)

    # Use the selection
    st.write(f"You selected: {llm_model}")
    st.header("Upload Transcripts")
    uploaded_files = st.file_uploader("Select TXT files", type=['txt'], accept_multiple_files=True)

    if uploaded_files and st.button("Initialize Processing"):
        if os.path.exists(BASE_DIR):
            shutil.rmtree(BASE_DIR)
        os.makedirs(DISCOVERY_DIR)

        total_files = list(uploaded_files)
        random.shuffle(total_files)

        # Requirement: 10 for discovery, rest in base
        discovery_files = total_files[:5]
        remaining_files = total_files[5:]

        for f in discovery_files:
            with open(os.path.join(DISCOVERY_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        for f in remaining_files:
            with open(os.path.join(BASE_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())

        # Clear previous state on new upload
        for key in ['discovery_df', 'key_mapping', 'df_selectable', 'df_display']:
            if key in st.session_state: del st.session_state[key]

        st.session_state.processed = True
        st.success(f"Split: 10 Discovery | {len(remaining_files)} Remaining")

# --- STEP 2: DISCOVERY & SELECTION ---
if st.session_state.processed and user_api_key:

    # Cache LLM and Discovery Results in Session State
    if 'discovery_df' not in st.session_state:
        with st.spinner("Analyzing Discovery Set (10 calls)..."):
            llm = ChatGroq(model=llm_model, api_key=user_api_key, temperature=0)


            # Internal Discovery Chain
            discovery_prompt = PromptTemplate(
                template="""
            You are an information extraction system.

            Task:
            Extract ALL factual details mentioned in the conversation below.
            Do NOT structure this as questions and answers.
            Do NOT infer or guess information.
            Only extract facts explicitly stated by either the member or the agent.

            Rules:
            - Capture details from BOTH member and agent
            - Normalize similar facts under consistent keys
            - Use clear, descriptive key names
            - If multiple values exist, return them as lists
            - Dates should be in ISO format if possible (YYYY-MM-DD)
            - Return ONLY valid JSON, no explanation

            Examples of facts to capture (not exhaustive):
            - member_name
            - agent_name
            - provider_name
            - hospital_name
            - visit_date
            - claim_amount
            - claim_status
            - denial_reason
            - department
            - issue_type
            - reference_numbers

            Conversation:
            {transcript}
            """,
                input_variables=["transcript"]
            )
            discovery_chain = discovery_prompt | llm | JsonOutputParser()

            # Process Transcripts
            transcripts = load_transcripts(DISCOVERY_DIR)
            extracted_facts = extract_facts(transcripts, discovery_chain)
            raw_df = build_fact_table(extracted_facts)
            raw_df = force_string_index(raw_df)

            # Map Keys
            mapping = generate_key_mapping(raw_df.index.tolist(), llm)

            # Prepare UI Dataframes
            st.session_state.discovery_df = raw_df
            st.session_state.key_mapping = mapping
            st.session_state.df_display = normalize_fact_dataframe(raw_df, mapping)

            sel_df = build_fact_sample_view(raw_df)
            sel_df["sample_values"] = sel_df["sample_values"].apply(lambda x: ", ".join(x))
            st.session_state.df_selectable = sel_df

    st.divider()
    tab_discovery, tab_results = st.tabs(["🔍 Discovery Selection", "📈 Analysis Results"])

    with tab_discovery:
        st.subheader("Select Rows for Further Analysis")

        # Ensure 'Select' column exists
        if "Select" not in st.session_state.df_selectable.columns:
            st.session_state.df_selectable.insert(0, "Select", False)

        edited_df = st.data_editor(
            st.session_state.df_selectable,
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
            disabled=[col for col in st.session_state.df_selectable.columns if col != "Select"],
            use_container_width=True,
            key="editor"
        )

        st.subheader("Reference Data (Normalized)")
        try :
            st.dataframe(st.session_state.df_display, use_container_width=True)
        except :
            st.write("Error with Normalized DF")

        if st.button("Run Final Analysis on Selected Keys"):
            selected_keys = edited_df[edited_df["Select"] == True]['fact_key'].tolist()

            if selected_keys:
                with st.spinner("Processing remaining transcripts..."):
                    llm = ChatGroq(model=llm_model, api_key=user_api_key, temperature=0)

                    entity_prompt = PromptTemplate(
                        template="""You are extracting factual entities from a call transcript.

                    Below is a fixed list of allowed entity keys.
                    You MUST ONLY use keys from this list.
                    Do NOT invent new keys.
                    Do NOT include keys that are not explicitly mentioned in the transcript.

                    ALLOWED ENTITY KEYS:
                    {entity_list}

                    INSTRUCTIONS:
                    - Extract ONLY entities that are clearly stated in the transcript
                    - If an entity is NOT mentioned, DO NOT include it in the output
                    - Do NOT infer, guess, or assume missing information
                    - Use the exact entity key names as provided
                    - If multiple distinct values exist for the same entity, return them as a list
                    - If only one value exists, return it as a string
                    - Values must be factual details stated by either the member or the agent
                    - Do NOT rephrase values unless necessary for clarity
                    - Do NOT include nulls, empty strings, or placeholder values

                    OUTPUT FORMAT RULES (MANDATORY):
                    - Return ONLY valid JSON
                    - No markdown
                    - No explanations
                    - No comments
                    - No trailing commas


                    TRANSCRIPT:
                    {transcript}""",
                        input_variables=["entity_list", "transcript"]
                    )
                    entity_chain = entity_prompt | llm | JsonOutputParser()

                    # Process Remaining (in base dir)
                    results = extract_controlled_entities(REMAINING_DIR, selected_keys, entity_chain)

                    st.session_state.df_final_1 = build_entity_table(results, selected_keys)
                    st.session_state.df_final_2 = st.session_state.df_final_1.copy()
                    st.session_state.final_step = True
                    st.success("Analysis Complete! Head to the Results tab.")
            else:
                st.warning("Please select at least one row.")

    # --- STEP 3: FINAL DISPLAY ---
    with tab_results:
        if st.session_state.final_step:
            st.write("### Final Analysis Results")
            st.dataframe(st.session_state.df_final_1)
            st.write(results)

            # Download Mechanism
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.df_final_1.to_excel(writer, sheet_name='Analysis_Results')

            st.download_button(
                label="📥 Download Results as Excel",
                data=output.getvalue(),
                file_name="transcript_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Results will appear here after selection in the Discovery tab.")
elif not user_api_key and st.session_state.processed:
    st.warning("Please enter your API key in the sidebar to begin processing.")
