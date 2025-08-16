import os
import csv
import json
import re
import threading
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import re
from sklearn.metrics import precision_recall_fscore_support, classification_report
import os
import pandas as pd
import time
import requests
from detoxify import Detoxify



PERSPECTIVE_URL = (
    "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
)
# Set which Perspective attributes to request:
PERSPECTIVE_ATTRIBUTES = {
    "TOXICITY": {},
    "IDENTITY_ATTACK": {}
}
load_dotenv()
CAD_PATH = os.getenv("CAD_V1_1_PATH", "../data/cad_v1_1.tsv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "outputs/detected_vs_true.csv")
API_KEYS = [
    "AIzaSyD9Oi-5LAHMTsyxKKXwRpUGcQhg93gsZwQ",
    "AIzaSyA50sg53qfozdDK7TIB_0xgWVw9zAvVV5Q",
    "AIzaSyAm9xPE02_bXhymfv-vti5AbYjfRZ4wTyo",
    "AIzaSyBdZcXLJS85eYPfOSMbTMGJUv1cABtrUtM",
    "AIzaSyBL1YUf976MThHkZoZxYawH_K2wypq07DE",
    "AIzaSyCTvYvBQKpt-CoApJmCkDyQxPcH95hnzUw"
]
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Primary and secondary labels will be loaded after reading the data
PRIMARY_LABELS: List[str] = ['AffiliationDirectedAbuse', 'CounterSpeech', 'IdentityDirectedAbuse', 'Neutral', 'PersonDirectedAbuse', 'Slur']
SECONDARY_LABELS: List[str] = ['NA', 'AffiliationDirectedAbuse / animosity', 'AffiliationDirectedAbuse / dehumanization', 'AffiliationDirectedAbuse / derogation', 'AffiliationDirectedAbuse / threatening language', 'CounterSpeech / affiliation-directed', 'CounterSpeech / identity-directed', 'CounterSpeech / person-directed', 'IdentityDirectedAbuse / animosity', 'IdentityDirectedAbuse / dehumanization', 'IdentityDirectedAbuse / derogation', 'IdentityDirectedAbuse / glorification', 'IdentityDirectedAbuse / threatening language', 'PersonDirectedAbuse / about a person who is not in the thread', 'PersonDirectedAbuse / directed to a person in the thread']

# Few-shot examples (trimmed for brevity)
FEW_SHOT = f"""
You are a highly trained moderator analyzing public conversations on platforms like Reddit, Twitter, and chat forums. Your task is to detect and explain any presence of:
Subtle Hate Speech: Implicit, indirect, or coded expressions of hostility or dehumanization toward people based on race, ethnicity, gender, sexuality, religion, nationality, disability, or other protected traits. Often appears as sarcasm, stereotypes, 'jokes', dog whistles, or “just asking questions” tactics.
Microaggressions: Dismissive, patronizing, or stereotypical remarks that reflect unconscious bias. These can be:
Microinsults (e.g., "You're so articulate for a ___"),
Microinvalidations (e.g., "I don’t see color"),
Assumptions based on identity (e.g., "Where are you really from?").
Counterspeech: Thoughtful or assertive responses that call out, deconstruct, or oppose hateful or biased content, either directly or through education, satire, or empathy.

Instructions:
Analyze the entire conversation, not just single posts. Pay attention to how context builds over multiple replies.
Watch for coded language, sarcasm, dog whistles, and euphemisms (e.g., "globalists", "tradwives", "based", "clown world").
Focus on the intent, implication, and social meaning behind the text.
Identify counterspeech that may be subtle, emotional, sarcastic, or deeply informed.

Allowed Primary labels: {PRIMARY_LABELS}
Allowed Secondary labels: {SECONDARY_LABELS}
Respond ONLY in JSON: {{ "primary_label":one of {PRIMARY_LABELS}, "secondary_label": one of {SECONDARY_LABELS}}}
Do NOT invent labels. If not hateful, primary_label must be "Neutral" and secondary_label "NA".

# Few-Shot Examples 
# AffiliationDirectedAbuse / animosity
tumblr makes me afraid to go outside in case i get too close to a feminist.
{{"primary_label": "AffiliationDirectedAbuse", "secondary_label": "AffiliationDirectedAbuse/animosity"}}

watching the left applaud her is like the popular kid in school making a joke that doesn't even make sense but everybody still laughs.
{{"primary_label": "AffiliationDirectedAbuse", "secondary_label": "AffiliationDirectedAbuse/animosityy"}}

# AffiliationDirectedAbuse / derogation
that sub, and any reddit youtube bait sub have mods that are absolutely cancer.
{{"primary_label": "AffiliationDirectedAbuse", "secondary_label": "AffiliationDirectedAbuse / derogation"}}

the mods are dictators. the end.
{{"primary_label": "AffiliationDirectedAbuse", "secondary_label": "AffiliationDirectedAbuse / derogation"}}

# AffiliationDirectedAbuse / dehumanization
most feminists are good people — most neo-nazis are good people... oh wait.
{{"primary_label": "AffiliationDirectedAbuse", "secondary_label": "AffiliationDirectedAbuse / dehumanizationn"}}

# AffiliationDirectedAbuse / threatening language
daughter: can you take me to school? me: sorry sweetie, i can't be alone with you, it wouldn't be *feminist* of me.
{{"primary_label": "AffiliationDirectedAbuse", "secondary_label": "AffiliationDirectedAbuse / threatening language"}}

# PersonDirectedAbuse / directed to person in the thread
you aren't the art police, cunt.
{{"primary_label": "PersonDirectedAbuse", "secondary_label": "PersonDirectedAbuse / directed to person in the thread"}}

you fucking wannabe serial killer.
{{"primary_label": "PersonDirectedAbuse", "secondary_label": "PersonDirectedAbuse / directed to person in the thread"}}

# PersonDirectedAbuse / about a person who is not in the thread
margaret thatcher, as in the woman who funded death squads in ireland, right?
{{"primary_label": "PersonDirectedAbuse", "secondary_label": "PersonDirectedAbuse / about a person who is not in the thread"}}

the new mod is garbage.
{{"primary_label": "PersonDirectedAbuse", "secondary_label": "PersonDirectedAbuse / about a person who is not in the thread"}}

# IdentityDirectedAbuse / derogation
ol' mikey was gay as all fuck. leo too i think?
{{"primary_label": "IdentityDirectedAbuse", "secondary_label": "IdentityDirectedAbuse / derogation"}}

i'll probably live a long life and die old and weak like a bitch.
{{"primary_label": "IdentityDirectedAbuse", "secondary_label": "IdentityDirectedAbuse / derogation"}}

# IdentityDirectedAbuse / animosity
you haven’t seen those vids where the entire neighborhoods are 100% muslims?
{{"primary_label": "IdentityDirectedAbuse", "secondary_label": "IdentityDirectedAbuse / animosity"}}

there are tons of muslim migrants in russia from central asia and from china.
{{"primary_label": "IdentityDirectedAbuse", "secondary_label": "IdentityDirectedAbuse / animosity"}}

# IdentityDirectedAbuse / glorification
butt pizza is a slur and it’s hilarious.
{{"primary_label": "IdentityDirectedAbuse", "secondary_label": "IdentityDirectedAbuse / glorification"}}

# IdentityDirectedAbuse / threatening language
she*male*
{{"primary_label": "IdentityDirectedAbuse", "secondary_label": "IdentityDirectedAbuse / threatening language"}}

# IdentityDirectedAbuse / dehumanization
picture an overweight middle-aged woman in an office saying it.
{{"primary_label": "IdentityDirectedAbuse", "secondary_label": "IdentityDirectedAbuse / dehumanization"}}

# CounterSpeech / identity-directed
trans women are women
{{"primary_label": "CounterSpeech", "secondary_label": "CounterSpeech / identity-directed"}}

so by your logic, the black people sold into slavery by fellow black people weren't oppressed?
{{"primary_label": "CounterSpeech", "secondary_label": "CounterSpeech / identity-directed"}}

# CounterSpeech / affiliation-directed
what the hell are you talking about?
{{"primary_label": "CounterSpeech", "secondary_label": "CounterSpeech / affiliation-directed"}}

it hurts to see people just generalizing cops like this while most of them are saving lives.
{{"primary_label": "CounterSpeech", "secondary_label": "CounterSpeech / affiliation-directed"}}

# CounterSpeech / person-directed
they're making fun of you. well, they are not doing a very good job.
{{"primary_label": "CounterSpeech", "secondary_label": "CounterSpeech / person-directed"}}

that's not well said. it's just trying to justify attacking people based on sexuality.
{{"primary_label": "CounterSpeech", "secondary_label": "CounterSpeech / person-directed"}}

# Slur
bitch
{{"primary_label": "Slur", "secondary_label": "NA"}}

nigga
{{"primary_label": "Slur", "secondary_label": "NA"}}

# Neutral
I went for a walk today.
{{"primary_label": "Neutral", "secondary_label": "NA"}}

What time is the meeting?
{{"primary_label": "Neutral", "secondary_label": "NA"}}

"""


def load_contextual_data(tsv_path, max_context=3):
    """Reads TSV and yields dicts with varying context lengths."""
    
    
    df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE,
                     keep_default_na=False, dtype=str)
    
    df = df.rename(columns={
        'id': 'post_id',
        'info_thread.id': 'thread_id',
        'info_order': 'order',
        'meta_text': 'utterance',
        'annotation_Context': 'context_str',
        'annotation_Primary': 'true_primary',
        'annotation_Secondary': 'true_secondary'
    })
    return df
def get_context_from_info_order(info_order: str, n: int, df: pd.DataFrame):
    """
    Given an info_order like '714,02' and a number n,
    return the last n turns from its conversation path (excluding parallel subtrees),
    always including title and post if present.
    """
    thread_id = info_order.split('-')[0].split(',')[0]

    thread_df = df[df["thread_id"] == thread_id].copy()

    order_map = {row["order"]: row for _, row in thread_df.iterrows()}

    context_orders = []
    if f"{thread_id}-title" in order_map:
        context_orders.append(f"{thread_id}-title")
    if f"{thread_id}-post" in order_map:
        context_orders.append(f"{thread_id}-post")

    if "-title" not in info_order and "-post" not in info_order:
        parts = info_order.split(",")
        for i in range(1, len(parts) + 1):
            ancestor = ",".join(parts[:i])
            if ancestor != info_order and ancestor in order_map:
                context_orders.append(ancestor)

    if info_order in order_map:
        context_orders.append(info_order)

    selected_orders = context_orders[-n:]

    context = [
        {
            "speaker": order_map[o]["meta_author"],
            "text": order_map[o]["utterance"]
        }
        for o in selected_orders if o in order_map
    ]
    return context


def detect_detoxify(text, model):

    return model.predict(text)

def detect_perspective(text, api_key):
    """Call Perspective API; return requested attribute scores or None on failure."""
    if not api_key:
        raise ValueError("Set PERSPECTIVE_API_KEY in your environment.")

    payload = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": PERSPECTIVE_ATTRIBUTES
    }
    params = {"key": api_key}

    try:
        resp = requests.post(PERSPECTIVE_URL, json=payload, params=params, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.warning(f"Perspective API returned {resp.status_code}: {resp.text}")
        # Return None or default zeros so the rest of the pipeline can continue
        return {attr.lower(): None for attr in PERSPECTIVE_ATTRIBUTES}
    except requests.exceptions.RequestException as e:
        logging.warning(f"Perspective request failed: {e}")
        return {attr.lower(): None for attr in PERSPECTIVE_ATTRIBUTES}

    data = resp.json()
    scores = {}
    for attr, info in data.get("attributeScores", {}).items():
        scores[attr.lower()] = info["summaryScore"]["value"]
    return scores
def run_all(df:pd.DataFrame,info_order:str, perp_key:str,c:int)->str:
     df = df.rename(columns={
        'id': 'post_id',
        'info_thread.id': 'thread_id',
        'info_order': 'order',
        'meta_text': 'utterance',
        'annotation_Context': 'context_str',
        'annotation_Primary': 'true_primary',
        'annotation_Secondary': 'true_secondary'
     })
     x =-1
     original=df.loc[df["order"] ==info_order, "utterance"].values
     context=""
     for i in range(c):
       context=get_context_from_info_order(info_order,i,df)
       res=classify(original,context)
       value=df.loc[df["order"] == info_order, "true_primary"].values
       if res['primary_label']==value[0]:
        x=i
        break
     orig_arr = df.loc[df["order"] == info_order, "utterance"].values
     if len(orig_arr) == 0:
      raise ValueError(f"info_order {info_order} not found in DataFrame.")
     orig = orig_arr[0]
     ctx_texts = []
     for turn in context:
    # turn might be {"speaker":..., "text":...}
      speaker = turn.get("speaker", "User")
      text_snip = turn.get("text", "")
      ctx_texts.append(f"{speaker}: {text_snip}")

# 3. Build full text for detox (single string)
     full_text = " ".join(ctx_texts + [orig])
     detox_model = Detoxify("original")
     detox_scores = detect_detoxify(full_text, detox_model)
     pers_scores = detect_perspective(full_text,perp_key)
     EXPLAIN=f"""
    You are a world-class content moderation expert, fluent in detecting nuanced hate speech, microaggressions, counterspeech, and toxicity in online threads. I will give you:
• The preceding conversation context: an ordered list of messages (“User1: …”, “User2: …”, etc.)
• The single current comment to analyze.
• A PrimaryLabel and SecondaryLabel from a hate-speech classifier:  
  – PRIMARY_LABELS = ['AffiliationDirectedAbuse', 'CounterSpeech', 'IdentityDirectedAbuse', 'Neutral', 'PersonDirectedAbuse', 'Slur']  
  – SECONDARY_LABELS = [
      'NA',
      'AffiliationDirectedAbuse / animosity',
      'AffiliationDirectedAbuse / dehumanization',
      'AffiliationDirectedAbuse / derogation',
      'AffiliationDirectedAbuse / threatening language',
      'CounterSpeech / affiliation-directed',
      'CounterSpeech / identity-directed',
      'CounterSpeech / person-directed',
      'IdentityDirectedAbuse / animosity',
      'IdentityDirectedAbuse / dehumanization',
      'IdentityDirectedAbuse / derogation',
      'IdentityDirectedAbuse / glorification',
      'IdentityDirectedAbuse / threatening language',
      'PersonDirectedAbuse / about a person who is not in the thread',
      'PersonDirectedAbuse / directed to a person in the thread'
    ]
• Toxicity metrics using Detoxify and Perspective API (refer to documentation for both)
 

Your job is to read all of that and produce a **plain-text report** with these four sections:

1. Overall Tone  
   - State “Negative”, “Neutral”, or “Positive” and in one or two sentences explain why (reference tone cues, hostility, empathy, etc.).

2. Label Validation 
   - Restate the original PrimaryLabel and SecondaryLabel.  
   - Say whether you agree or disagree.  
   - If you disagree, propose the corrected labels.  
   - Explain your reasoning by pointing to specific words, intent, or context.

3. Toxicity Analysis  
   - Restate the Detoxify and Perspective scores and explain them in short.
   - Declare whether the comment (or context) is toxic.  
   - Provide a short natural-language summary of what kind of toxicity is present (e.g. “dehumanizing language”, “threat of violence”, “sarcastic invalidation”).  
   - Quote each toxic phrase or sentence, and explain immediately after why it’s toxic.

4. Non-Toxic Insights  
   - Summarize in one or two sentences any constructive, informative, or neutral points made in the thread, excluding all toxic material.

Be sure to consider sarcasm, dog whistles, backhanded compliments, and how the context changes meaning. Use the toxicity scores as guidance (e.g. treat >0.7 as likely toxic) but rely on your expert judgment.  
Now for the data
The text is{original}
The context is{context}
The classifier labels are {res}
The toxicity scores are {detox_scores}
The perspective scores are{pers_scores}
Now, produce your analysis in plain text under the four headings above. Do not output JSON—just clear, human-readable sections.

     """
     key = get_next_key()
     configure_api(key)
     prompt = EXPLAIN 
     chat = genai.GenerativeModel(MODEL).start_chat()
     response = chat.send_message(prompt)
     text = response.text
     ans1="Min turns required for detecting hate"+str(x)
     text=ans1+text

     return text

_key_lock = threading.Lock()
_key_index = 0

def configure_api(key: str) -> None:
    """Configure the genai library with the provided API key."""
    genai.configure(api_key=key)


def get_next_key() -> str:
    """Rotate through API keys in a thread-safe manner."""
    global _key_index
    with _key_lock:
        key = API_KEYS[_key_index % len(API_KEYS)]
        _key_index += 1
    return key


def parse_order_key(o: str) -> List[int]:
    parts = re.split(r"[,-]", o)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        elif p == 'title':
            key.append(-2)
        elif p == 'post':
            key.append(-1)
        else:
            key.append(-3)
    return key


def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess the CAD dataset."""
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE,
                     keep_default_na=False, dtype=str)
    if 'split' in df.columns:
        df = df[df['split'] != 'exclude_empty']
    df = df.rename(columns={
        'id': 'post_id',
        'info_thread.id': 'thread_id',
        'info_order': 'order',
        'meta_text': 'utterance',
        'annotation_Context': 'context_str',
        'annotation_Primary': 'true_primary',
        'annotation_Secondary': 'true_secondary'
    });df = df[df["split"] == "test"].reset_index(drop=True)
    return df


def build_threads(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group posts by thread and sort by their hierarchical order."""
    threads: Dict[str, pd.DataFrame] = {}
    for tid, group in df.groupby('thread_id', sort=False):
        g = group.copy()
        g['order_key'] = g['order'].apply(parse_order_key)
        g = g.sort_values('order_key').reset_index(drop=True)
        threads[tid] = g
    return threads


def build_thread_contexts(threads: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, List[str]]]:
    """Precompute ancestor utterances for each post in each thread."""
    contexts: Dict[str, Dict[str, List[str]]] = {}
    for tid, tdf in threads.items():
        orders = tdf['order'].tolist()
        utts = tdf['utterance'].tolist()
        order2utt = dict(zip(orders, utts))
        ctx_map: Dict[str, List[str]] = {}
        for o in orders:
            ancestors = [p for p in orders if o.startswith(p) and p != o]
            ancestors.sort(key=lambda p: p.count(',') + p.count('-'))
            ctx_map[o] = [order2utt[p] for p in ancestors]
        contexts[tid] = ctx_map
    return contexts

def classify(utterance: str, context: List[str]) -> Dict[str, Any]:
    """Call the LLM to classify a single utterance with its context."""
    key = get_next_key()
    configure_api(key)
    prompt = FEW_SHOT + "\nContext:\n" + "\n".join(f"- {c}" for c in context)
    prompt += f"\nMessage: \"{utterance}\"\n"
    chat = genai.GenerativeModel(MODEL).start_chat()
    response = chat.send_message(prompt)
    text = response.text;print(text)
    match = re.search(r'(?s)\{.*?\}', text)
    if match:
        text = match.group(0)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"primary_label": "Neutral", "secondary_label": "NA"};print("json did not load");
    result['is_hate'] = bool(result.get('is_hate', False))
    if result['primary_label'] not in PRIMARY_LABELS:
        result['primary_label'] = 'Neutral'
    if result['secondary_label'] not in SECONDARY_LABELS:
        result['secondary_label'] = 'NA'
    return result

def run_classification(threads: Dict[str, pd.DataFrame],
                       contexts: Dict[str, Dict[str, List[str]]],
                       output_csv: str) -> List[Dict[str, str]]:
    """Process every post sequentially, write results to CSV, and collect for evaluation."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results: List[Dict[str, str]] = []
    with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(["post_id","thread_id","order",
                         "true_primary","true_secondary",
                         "pred_primary","pred_secondary"])
        for tid, tdf in threads.items():
            for _, row in tdf.iterrows():
                pid = row['post_id'];print(pid)
                order = row['order']
                utt = row['utterance']
                true_p = row['true_primary']
                true_s = row['true_secondary'] or 'None'
                ctx = contexts[tid][order]
                res = classify(utt, ctx)
                writer.writerow([pid, tid, order, true_p,
                                 true_s, res['primary_label'],
                                 res['secondary_label']])
                results.append({
                    'true_primary': true_p,
                    'pred_primary': res['primary_label']
                })
    return results


def evaluate(results: List[Dict[str, str]]) -> None:
    """Compute and print macro precision, recall, F1 and a full classification report."""
    y_true = [r['true_primary'] for r in results]
    y_pred = [r['pred_primary'] for r in results]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=PRIMARY_LABELS, average='macro')
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}\n")
    print(classification_report(y_true, y_pred, labels=PRIMARY_LABELS))


def main() -> None:
    # Test API connectivity
    if not API_KEYS:
        print("? No API keys found. Exiting.")
        return
    configure_api(API_KEYS[0])
    chat = genai.GenerativeModel(MODEL).start_chat()
    resp = chat.send_message("Hi")
    print("API test successful. Response:", resp.text)

    # Load data and prepare labels
    df = load_data(CAD_PATH)
    global PRIMARY_LABELS, SECONDARY_LABELS
    PRIMARY_LABELS = sorted(df['true_primary'].unique())
    print(PRIMARY_LABELS)
    SECONDARY_LABELS = sorted([lbl or 'NA' for lbl in df['true_secondary'].fillna('').unique()])
    print(SECONDARY_LABELS)
    
    threads = build_threads(df)
    contexts = build_thread_contexts(threads)
    results = run_classification(threads, contexts, OUTPUT_CSV)
    evaluate(results)


if __name__ == "__main__":
    main()
