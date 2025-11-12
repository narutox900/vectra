import json
from datetime import datetime

import google.generativeai as genai
import openai
import pandas as pd
import requests
import streamlit as st

try:
    from serpapi import GoogleSearch
except ImportError:  # SerpAPI is optional
    GoogleSearch = None

# App config
st.set_page_config(page_title="Vectra", layout="wide")
st.title("🔍 Vectra: Query Fan-Out Simulator for AI Surfaces")

# Sidebar: API key input and query
st.sidebar.header("Configuration")
run_clicked = st.sidebar.button("Run Fan-Out 🚀")
provider = st.sidebar.radio("AI Provider", ["Gemini", "OpenAI"])

if provider == "Gemini":
    api_key = st.sidebar.text_input("Gemini API Key", type="password", key="gemini_key")
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_key")

input_mode = st.sidebar.radio("Input Mode", ["Single query", "Bulk list"])
if input_mode == "Single query":
    user_query = st.sidebar.text_area(
        "Enter your query",
        "What's the best electric SUV for driving up mt rainier?",
        height=120
    )
else:
    bulk_text = st.sidebar.text_area(
        "Paste queries (one per line)",
        "best electric suv for snow\nsleep training methods for toddlers\nhow to freeze sourdough starter",
        height=180
    )

mode = st.sidebar.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])

# Configure AI provider
api_key_missing = not api_key

if provider == "Gemini":
    # You can change to a pinned version like "gemini-2.5-pro-exp-0827" if desired.
    model_name = "gemini-2.5-pro"
    model = None
    ai_client = None
    if not api_key_missing:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
else:  # OpenAI
    model_name = "gpt-4o"  # Using GPT-4o as default, can be changed
    model = None
    ai_client = openai.OpenAI(api_key=api_key) if not api_key_missing else None

st.sidebar.markdown("---")
st.sidebar.subheader("SerpAPI (optional)")
serpapi_key = st.sidebar.text_input("SerpAPI Key", type="password")
fetch_serp_results = st.sidebar.checkbox(
    "Fetch Google AI Overview for each fan-out query",
    value=False,
    help="Calls SerpAPI's Google AI Overview API for every generated query."
)
serpapi_location = st.sidebar.text_input(
    "Location",
    value="Austin, Texas, United States",
    help="Passed to SerpAPI as `location`.",
    disabled=not fetch_serp_results
)
serpapi_domain = st.sidebar.text_input(
    "Google domain",
    value="google.com",
    help="Passed as `google_domain` (e.g., google.com, google.co.uk).",
    disabled=not fetch_serp_results
)
serpapi_gl = st.sidebar.text_input(
    "Country code (gl)",
    value="us",
    help="Two-letter country code sent as `gl`.",
    disabled=not fetch_serp_results
)
serpapi_hl = st.sidebar.text_input(
    "Language (hl)",
    value="en",
    help="Interface language sent as `hl`.",
    disabled=not fetch_serp_results
)

if fetch_serp_results and not serpapi_key:
    st.sidebar.warning("Provide a SerpAPI key or disable fetching Google results.")

# Allowed routing formats (sent to the model)
ALLOWED_FORMATS = [
    "web_article",
    "faq_page",
    "how_to_steps",
    "comparison_table",
    "buyers_guide",
    "checklist",
    "product_spec_sheet",
    "glossary/definition",
    "pricing_page",
    "review_roundup",
    "tutorial_video/transcript",
    "podcast_transcript",
    "code_samples/docs",
    "api_reference",
    "calculator/tool",
    "dataset",
    "image_gallery",
    "map/local_pack",
    "forum/qna",
    "pdf_whitepaper",
    "case_study",
    "press_release",
    "interactive_widget"
]

# Prompt builder
def QUERY_FANOUT_PROMPT(q, mode):
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"you must decide on an optimal number of queries to generate. "
            f"This number must be at least {min_queries_simple}. "
            f"For a straightforward query, generate around {min_queries_simple}-{min_queries_simple + 2}. "
            f"If the query has a few distinct aspects or common follow-ups, aim for {min_queries_simple + 3}-{min_queries_simple + 5}. "
            f"Provide brief reasoning for why you chose this number."
        )
    else:
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"you must decide on an optimal number of queries to generate. "
            f"This number must be at least {min_queries_complex}. "
            f"For multifaceted queries that span comparisons, procedures, specs, or trade-offs, "
            f"generate {min_queries_complex + 5}-{min_queries_complex + 10} or more. "
            f"Provide brief reasoning for your number."
        )

    routing_note = (
        "For EACH expanded query, also identify the most likely CONTENT TYPE / FORMAT the routing system would prefer "
        "for retrieval and synthesis (e.g., a how-to should route to 'how_to_steps' or a video transcript; comparisons to 'comparison_table' or 'buyers_guide'). "
        "Choose exactly ONE label from this fixed list:\n"
        + ", ".join(ALLOWED_FORMATS) +
        ".\nReturn it in a field named 'routing_format' and give a short 'format_reason' (1 sentence)."
    )

    return (
        f"You are simulating Google's AI Mode query fan-out for generative search systems.\n"
        f"The user's original query is: \"{q}\". The selected mode is: \"{mode}\".\n\n"
        f"Your first task is to determine the total number of queries to generate and the reasoning for this number:\n"
        f"{num_queries_instruction}\n\n"
        f"Once you have decided on the number and the reasoning, generate exactly that many unique synthetic queries.\n"
        f"Each of the following transformation types MUST be represented at least once, if the total allows:\n"
        f"1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n5. Entity Expansions\n6. Personalized Queries\n\n"
        f"The 'reasoning' field for each query should explain why that query was generated (tie it to the original query, its type, and user intent). "
        f"Do NOT include queries dependent on real-time user history or geolocation.\n\n"
        f"{routing_note}\n\n"
        f"Return only a valid JSON object in this exact schema:\n"
        "{\n"
        "  \"generation_details\": {\n"
        "    \"target_query_count\": 12,\n"
        "    \"reasoning_for_count\": \"...\"\n"
        "  },\n"
        "  \"expanded_queries\": [\n"
        "    {\n"
        "      \"query\": \"...\",\n"
        "      \"type\": \"reformulation | related | implicit | comparative | entity_expansion | personalized\",\n"
        "      \"user_intent\": \"...\",\n"
        "      \"reasoning\": \"...\",\n"
        "      \"routing_format\": \"one_of_allowed_labels\",\n"
        "      \"format_reason\": \"one sentence why this format is best\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

def maybe_expand_ai_overview(results, api_key):
    """Fetch multi-page AI overview details when SerpAPI indicates an extra request."""
    ai_overview = results.get("ai_overview") or {}
    serpapi_link = ai_overview.get("serpapi_link")
    page_token = ai_overview.get("page_token")
    if not serpapi_link or not page_token:
        return results, None

    try:
        print("enriching")
        response = requests.get(
            serpapi_link,
            params={"api_key": api_key, "page_token": page_token},
            timeout=90
        )
        response.raise_for_status()
        payload = response.json()
        enriched = payload.get("ai_overview") if isinstance(payload, dict) else None
        if isinstance(enriched, dict):
            results["ai_overview"] = enriched
        return results, None
    except Exception as exc:
        return results, f"AI Overview continuation failed: {exc}"


def get_serpapi_results(
    query_text,
    api_key,
    location=None,
    google_domain=None,
    gl=None,
    hl=None
):
    """Retrieve full SerpAPI response for a query."""
    if GoogleSearch is None:
        return [], "SerpAPI client library is not available. Please install google-search-results."
    if not query_text:
        return [], "Query is empty."

    try:
        params = {
            "engine": "google",
            "q": query_text,
            "api_key": api_key
        }
        if location:
            params["location"] = location
        if google_domain:
            params["google_domain"] = google_domain
        if gl:
            params["gl"] = gl
        if hl:
            params["hl"] = hl
        search = GoogleSearch(params)
        results = search.get_dict()
        ai_overview_error = None
        print(results)
        if isinstance(results.get("ai_overview"), dict) and results.get("ai_overview").get("page_token") and results.get("ai_overview").get("serpapi_link"):
            results, ai_overview_error = maybe_expand_ai_overview(results, api_key)
        return results, ai_overview_error
    except Exception as exc:
        return [], str(exc)

# Single fan-out
def generate_fanout(query, mode, provider_name, model_instance=None, openai_client=None, openai_model_name=None):
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    
    if provider_name == "Gemini":
        response = model_instance.generate_content(prompt)
        json_text = response.text.strip()
    else:  # OpenAI
        try:
            response = openai_client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates JSON responses for query fan-out simulation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            json_text = response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    # Clean code fences if present
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    if json_text.startswith("```"):
        json_text = json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    json_text = json_text.strip()

    data = json.loads(json_text)
    generation_details = data.get("generation_details", {})
    expanded_queries = data.get("expanded_queries", [])

    return generation_details, expanded_queries, json_text

# Initialize session state
if 'last_runs' not in st.session_state:
    st.session_state.last_runs = []

run_tab, analyze_tab = st.tabs(["Run Fan-Out", "Analyze Saved Runs"])

with run_tab:
    # Run button
    if run_clicked:
        if api_key_missing:
            st.error(f"Please enter your {provider} API Key to run the fan-out.")
        else:
            # Build list of lookup queries
            if input_mode == "Single query":
                lookups = [user_query.strip()] if user_query.strip() else []
            else:
                lookups = [q.strip() for q in bulk_text.splitlines() if q.strip()]

            if not lookups:
                st.warning("⚠️ Please provide at least one query.")
                st.stop()

            all_rows = []
            run_summaries = []
            errors = []
            raw_outputs = {}

            status = st.status("Processing queries…", expanded=True)
            progress = st.progress(0)
            total = len(lookups)

            for i, q in enumerate(lookups, start=1):
                try:
                    details, expanded, raw = generate_fanout(
                        q, mode, provider, 
                        model_instance=model if provider == "Gemini" else None,
                        openai_client=ai_client if provider == "OpenAI" else None,
                        openai_model_name=model_name if provider == "OpenAI" else None
                    )
                    raw_outputs[q] = raw
                    run_summaries.append({
                        "lookup_query": q,
                        "target_query_count": details.get("target_query_count"),
                        "reasoning_for_count": details.get("reasoning_for_count", "")
                    })
                    # Flatten rows, prefix with lookup query
                    for obj in expanded:
                        row = {
                            "lookup_query": q,
                            "query": obj.get("query", ""),
                            "type": obj.get("type", ""),
                            "user_intent": obj.get("user_intent", ""),
                            "reasoning": obj.get("reasoning", ""),
                            "routing_format": obj.get("routing_format", ""),
                            "format_reason": obj.get("format_reason", "")
                        }
                        if fetch_serp_results and serpapi_key:
                            serp_results, serp_error = get_serpapi_results(
                                row["query"],
                                serpapi_key,
                                location=serpapi_location,
                                google_domain=serpapi_domain,
                                gl=serpapi_gl,
                                hl=serpapi_hl
                            )
                            row["serpapi_results"] = json.dumps(serp_results, ensure_ascii=False)
                            row["serpapi_error"] = serp_error
                        all_rows.append(row)
                    # Add original lookup row for downstream analysis
                    original_row = {
                        "lookup_query": q,
                        "query": q,
                        "type": "original_lookup",
                        "user_intent": "Original lookup query",
                        "reasoning": "User-entered query before fan-out.",
                        "routing_format": "",
                        "format_reason": ""
                    }
                    if fetch_serp_results and serpapi_key:
                        serp_results, serp_error = get_serpapi_results(
                            q,
                            serpapi_key,
                            location=serpapi_location,
                            google_domain=serpapi_domain,
                            gl=serpapi_gl,
                            hl=serpapi_hl
                        )
                        original_row["serpapi_results"] = json.dumps(serp_results, ensure_ascii=False)
                        original_row["serpapi_error"] = serp_error
                    all_rows.append(original_row)
                    status.write(f"✅ Processed: **{q}** — generated {len(expanded)} queries.")
                except json.JSONDecodeError as e:
                    msg = f"❌ JSON parse failed for '{q}': {e}"
                    status.write(msg)
                    errors.append({"lookup_query": q, "error": str(e)})
                except Exception as e:
                    msg = f"❌ Error for '{q}': {e}"
                    status.write(msg)
                    errors.append({"lookup_query": q, "error": str(e)})

                progress.progress(i / total)

            status.update(label="Complete.", state="complete")

            # Build output DataFrame (lookup_query first)
            if all_rows:
                df = pd.DataFrame(all_rows)

                run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                run_id = f"{run_timestamp.replace(' ', '_').replace(':', '-')}_{len(st.session_state.last_runs) + 1}"
                df["run_id"] = run_id
                df["run_timestamp"] = run_timestamp
                df["provider"] = provider
                df["mode"] = mode
                df["input_mode"] = input_mode
                df["serpapi_enabled"] = fetch_serp_results and bool(serpapi_key)
                df["serpapi_location"] = serpapi_location
                df["serpapi_domain"] = serpapi_domain
                df["serpapi_gl"] = serpapi_gl
                df["serpapi_hl"] = serpapi_hl

                # Ensure column order (lookup_query first, metadata grouped)
                preferred_cols = [
                    "run_id",
                    "run_timestamp",
                    "provider",
                    "mode",
                    "input_mode",
                    "lookup_query",
                    "query",
                    "type",
                    "user_intent",
                    "reasoning",
                    "routing_format",
                    "format_reason",
                    "serpapi_enabled",
                    "serpapi_location",
                    "serpapi_domain",
                    "serpapi_gl",
                    "serpapi_hl"
                ]
                existing = [c for c in preferred_cols if c in df.columns]
                others = [c for c in df.columns if c not in existing]
                df = df[existing + others]

                df_display = df.copy()
                if fetch_serp_results and serpapi_key and "serpapi_results" in df_display.columns:
                    df_display["serpapi_results"] = df_display["serpapi_results"].fillna("")

                st.subheader("📊 Synthetic Queries (with routing format)")
                st.dataframe(df_display, use_container_width=True, height=(min(len(df_display), 20) + 1) * 35 + 3)

                csv = df_display.to_csv(index=False).encode("utf-8")
                csv_name = f"vectra_output_bulk_with_routing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button("📥 Download CSV", data=csv, file_name=csv_name, mime="text/csv")

                st.session_state.last_runs.append({
                    "timestamp": run_timestamp,
                    "provider": provider,
                    "mode": mode,
                    "input_mode": input_mode,
                    "lookup_queries": lookups,
                    "rows": df.to_dict(orient="records"),
                    "run_summaries": run_summaries,
                    "errors": errors,
                    "serpapi_enabled": fetch_serp_results and bool(serpapi_key),
                    "serpapi_location": serpapi_location,
                    "serpapi_domain": serpapi_domain,
                    "serpapi_gl": serpapi_gl,
                    "serpapi_hl": serpapi_hl,
                    "raw_outputs": raw_outputs
                })

                if fetch_serp_results and serpapi_key and "serpapi_results" in df.columns:
                    st.markdown("---")
                    st.subheader("🧠 Google AI Overview (SerpAPI)")
                    selectable_queries = df["query"].tolist()
                    selected_query = st.selectbox(
                        "Select a fan-out query to inspect AI Overview",
                        selectable_queries
                    )
                    selected_series = df[df["query"] == selected_query].iloc[0]
                    serp_results = selected_series.get("serpapi_results")
                    if isinstance(serp_results, str) and serp_results.strip():
                        try:
                            serp_results = json.loads(serp_results)
                        except json.JSONDecodeError:
                            pass
                    serp_error = selected_series.get("serpapi_error")

                    if serp_error:
                        st.warning(f"SerpAPI error for this query: {serp_error}")

                    overview = None
                    if isinstance(serp_results, dict):
                        overview = serp_results.get("ai_overview")

                    if overview:
                        st.json(overview)
                        references = overview.get("references")
                        if isinstance(references, list) and references:
                            ref_rows = []
                            for ref in references:
                                if isinstance(ref, dict):
                                    ref_rows.append({
                                        "title": ref.get("title"),
                                        "link": ref.get("link"),
                                        "source": ref.get("source"),
                                        "snippet": ref.get("snippet"),
                                    })
                            if ref_rows:
                                with st.expander("AI Overview References"):
                                    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True)
                    else:
                        st.info("No AI overview available.")
            else:
                st.warning("No synthetic queries were generated.")

            # Summaries per lookup (optional)
            if run_summaries:
                st.markdown("---")
                st.subheader("🧠 Generation Plans (per lookup)")
                sum_df = pd.DataFrame(run_summaries)
                st.dataframe(sum_df, use_container_width=True)

            # Error table if any
            if errors:
                st.markdown("---")
                st.subheader("⚠️ Errors")
                err_df = pd.DataFrame(errors)
                st.dataframe(err_df, use_container_width=True)

    # Saved runs viewer
    if st.session_state.get('last_runs'):
        st.markdown("---")
        st.subheader("💾 Saved Runs")
        runs = st.session_state.last_runs
        run_index = st.selectbox(
            "Select a saved run to review",
            options=range(len(runs)),
            format_func=lambda idx: f"{idx + 1}. {runs[idx]['timestamp']} | {runs[idx]['provider']} | {runs[idx]['mode']} | {len(runs[idx]['rows'])} rows",
            key="saved_run_selector"
        )
        saved_run = runs[run_index]
        saved_df = pd.DataFrame(saved_run["rows"]) if saved_run["rows"] else pd.DataFrame()
        saved_df_display = saved_df.copy()
        if "serpapi_results" in saved_df_display.columns:
            saved_df_display["serpapi_results"] = saved_df_display["serpapi_results"].fillna("")
        st.markdown(
            f"**Metadata:** Provider `{saved_run['provider']}`, Mode `{saved_run['mode']}`, "
            f"{'SerpAPI on' if saved_run.get('serpapi_enabled') else 'SerpAPI off'}"
        )
        if saved_run.get("serpapi_enabled"):
            st.caption(
                f"SerpAPI params → location: {saved_run.get('serpapi_location') or 'n/a'}, "
                f"domain: {saved_run.get('serpapi_domain') or 'n/a'}, "
                f"gl: {saved_run.get('serpapi_gl') or 'n/a'}, "
                f"hl: {saved_run.get('serpapi_hl') or 'n/a'}"
            )
        st.dataframe(
            saved_df_display,
            use_container_width=True,
            height=(min(len(saved_df_display), 20) + 1) * 35 + 3 if not saved_df_display.empty else 200
        )
        csv_name = f"vectra_saved_{saved_run['timestamp'].replace(' ', '_').replace(':', '-')}.csv"
        st.download_button(
            "📥 Download Saved Run CSV",
            data=saved_df_display.to_csv(index=False).encode("utf-8"),
            file_name=csv_name,
            mime="text/csv",
            key=f"download_saved_{run_index}"
        )
        if saved_run.get("run_summaries"):
            with st.expander("Generation plans for this run"):
                st.dataframe(pd.DataFrame(saved_run["run_summaries"]), use_container_width=True)
        if saved_run.get("errors"):
            with st.expander("Errors captured during this run"):
                st.dataframe(pd.DataFrame(saved_run["errors"]), use_container_width=True)
        if saved_run.get("raw_outputs"):
            with st.expander("Raw model JSON per lookup"):
                for lookup, raw_json in saved_run["raw_outputs"].items():
                    st.markdown(f"**Lookup:** {lookup}")
                    st.code(raw_json, language="json")

with analyze_tab:
    st.markdown("Upload one or more CSV exports (e.g., Saved Runs) to inspect them without re-running expensive queries.")
    uploaded_csvs = st.file_uploader(
        "Saved run CSV files",
        type=["csv"],
        key="analysis_uploader",
        accept_multiple_files=True
    )
    analysis_df = None
    if uploaded_csvs:
        frames = []
        for file in uploaded_csvs:
            try:
                frames.append(pd.read_csv(file))
            except Exception as exc:
                st.error(f"Could not read {file.name}: {exc}")
        if frames:
            analysis_df = pd.concat(frames, ignore_index=True)

    if analysis_df is not None:
            st.success("Saved run loaded.")
            cols = st.columns(3)
            cols[0].metric("Rows", len(analysis_df))
            if "lookup_query" in analysis_df.columns:
                cols[1].metric("Lookup queries", analysis_df["lookup_query"].nunique())
            if "type" in analysis_df.columns:
                cols[2].metric("Distinct types", analysis_df["type"].nunique())

            st.dataframe(
                analysis_df,
                use_container_width=True,
                height=(min(len(analysis_df), 25) + 1) * 35 + 3
            )

            serp_col = "serpapi_results"
            serp_data = None
            if serp_col in analysis_df.columns:
                def parse_serp(value):
                    if isinstance(value, str) and value.strip():
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            return value
                    return value

                serp_data = analysis_df[serp_col].apply(parse_serp)

            all_reference_rows = []
            if serp_data is not None and "query" in analysis_df.columns:
                query_list = analysis_df["query"].tolist()
                type_list = analysis_df["type"].tolist() if "type" in analysis_df.columns else [None] * len(query_list)
                lookup_list = analysis_df["lookup_query"].tolist() if "lookup_query" in analysis_df.columns else [None] * len(query_list)
                for idx, query in enumerate(query_list):
                    payload = serp_data.iloc[idx]
                    if isinstance(payload, dict):
                        overview = payload.get("ai_overview")
                        references = overview.get("references") if isinstance(overview, dict) else None
                        if isinstance(references, list):
                            for ref in references:
                                if isinstance(ref, dict):
                                    all_reference_rows.append({
                                        "lookup_query": lookup_list[idx],
                                        "query": query,
                                        "type": type_list[idx],
                                        "title": ref.get("title"),
                                        "link": ref.get("link"),
                                        "source": ref.get("source"),
                                        "snippet": ref.get("snippet")
                                    })

            if "query" in analysis_df.columns and serp_data is not None:
                st.markdown("---")
                st.subheader("SerpAPI AI Overview from uploaded run")
                available_queries = analysis_df["query"].tolist()
                query_choice = st.selectbox(
                    "Pick a fan-out query",
                    available_queries,
                    key="analysis_query_select"
                )
                selected_row = analysis_df[analysis_df["query"] == query_choice].iloc[0]
                selected_serp = serp_data.iloc[selected_row.name]

                overview = None
                if isinstance(selected_serp, dict):
                    overview = selected_serp.get("ai_overview")

                if overview:
                    st.json(overview)
                    references = overview.get("references")
                    if isinstance(references, list) and references:
                        ref_rows = []
                        for ref in references:
                            if isinstance(ref, dict):
                                ref_rows.append({
                                    "title": ref.get("title"),
                                    "link": ref.get("link"),
                                    "source": ref.get("source"),
                                "snippet": ref.get("snippet"),
                            })
                        if ref_rows:
                            with st.expander("AI Overview References"):
                                ref_df = pd.DataFrame(ref_rows)
                                st.dataframe(ref_df, use_container_width=True)
                else:
                    st.info("No AI overview available.")

            if all_reference_rows:
                st.markdown("---")
                st.subheader("AI Overview References (all queries)")
                filter_options = sorted({row["lookup_query"] for row in all_reference_rows if row.get("type") == "original_lookup" and row.get("lookup_query")})
                selected_originals = st.multiselect(
                    "Filter by original query",
                    filter_options,
                    default=filter_options,
                    key="analysis_original_filter"
                )
                all_ref_df = pd.DataFrame(all_reference_rows)
                if selected_originals:
                    all_ref_df = all_ref_df[all_ref_df["lookup_query"].isin(selected_originals)]
                original_entries = all_ref_df[all_ref_df["type"] == "original_lookup"] if "type" in all_ref_df.columns else pd.DataFrame()
                if not original_entries.empty:
                    originals = sorted({val for val in (original_entries["lookup_query"].dropna().tolist() + original_entries["query"].dropna().tolist()) if val})
                    if originals:
                        st.markdown("**Original queries:** " + ", ".join(f"`{q}`" for q in originals))
                if "type" in all_ref_df.columns and not all_ref_df["type"].isna().all():
                    all_ref_df["__is_original_lookup"] = all_ref_df["type"] == "original_lookup"
                    all_ref_df = all_ref_df.sort_values(by="__is_original_lookup", ascending=False).drop(columns="__is_original_lookup")
                st.dataframe(all_ref_df, use_container_width=True)
                if "source" in all_ref_df.columns:
                    source_counts = all_ref_df["source"].value_counts().reset_index()
                    source_counts.columns = ["source", "count"]
                    st.subheader("References by source")
                    st.dataframe(source_counts, use_container_width=True)

            if "type" in analysis_df.columns:
                st.markdown("---")
                st.subheader("Query type breakdown")
                type_counts = analysis_df["type"].value_counts().reset_index()
                type_counts.columns = ["type", "count"]
                st.bar_chart(type_counts.set_index("type"))
                st.subheader("Original query filter for type breakdown")
                if "lookup_query" in analysis_df.columns:
                    orig_options = sorted(analysis_df["lookup_query"].dropna().unique().tolist())
                    selected_origins = st.multiselect(
                        "Select original queries",
                        orig_options,
                        default=orig_options,
                        key="analysis_type_filter"
                    )
                    if selected_origins:
                        filtered = analysis_df[analysis_df["lookup_query"].isin(selected_origins)]
                        type_counts_filtered = filtered["type"].value_counts().reset_index()
                        type_counts_filtered.columns = ["type", "count"]
                        st.bar_chart(type_counts_filtered.set_index("type"))
