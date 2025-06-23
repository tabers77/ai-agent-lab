from langchain_community.tools import TavilySearchResults
from conf.configs import Cfg
from duckduckgo_search import DDGS

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from typing import List, Dict

from langchain.tools import tool
from utils import llm

configs_ = Cfg()
connection_string = configs_.database_configs.pg_connection_string

tool_search = TavilySearchResults(max_results=5)


@tool
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."


@tool
def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."


@tool
def web_search(query: str, num_results: int = 5) -> list[str]:  # Use 1 for testing , 5 default
    """Run a web search using DuckDuckGo (v8+ API) and return a list of Title - URL strings."""
    print('***DEBUG***: Running websearch with query:', query)
    # results = []
    results: List[Dict[str, str]] = []  # TEST
    # DDGS.text yields dicts with 'title' and 'href'
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=num_results):
            # title = hit.get("title", "No title")
            # url = hit.get("href", hit.get("url", ""))
            # results.append(f"{title} - {url}")
            # ---------------------------------------------
            # TEST
            title = hit.get("title", "No title").strip()
            url = hit.get("href", hit.get("url", "")).strip()
            results.append({"title": title, "url": url})
            # ---------------------------------------------

    #print('***DEBUG***: Web search results:', results)
    return results


@tool
def fetch_and_summarize(url: str, max_length: int = 300) -> str:
    """
    Fetch the content of a URL and return a concise summary using an LLM.

    Returns:
        str: The generated summary as a string.
    """

    print("***DEBUG***: Running fetch_and_summarize", fetch_and_summarize)

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching URL {url}: {e}"

    # Extract visible text from paragraphs
    soup = BeautifulSoup(resp.text, 'html.parser')
    paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
    text = ' '.join(paragraphs)
    if not text:
        return f"No extractable text found at {url}."

    # Truncate to a reasonable character limit for the LLM
    excerpt = text[:max_length * 10]

    # Build prompt
    template = (
        "You are an AI assistant that summarizes web articles.\n"
        "Please produce a concise summary (max {max_length} tokens) of the following text:\n\n"
        "{content}\n\n"
        "Return only the summary, without commentary."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", template)
    ])
    # Format and invoke
    messages = [HumanMessage(content=prompt.format(content=excerpt, max_length=max_length))]
    summary = llm.invoke(messages).content
   #print('***DEBUG***: Summary of fetch_and_summarize', summary)

    assert len(summary) > 0, "LLM returned an empty summary"

    return summary


def _build_article_prompt(topic: str, research_summaries: list[str], audience: str, tone: str) -> str:
    """
    Construct a prompt for the LLM that asks for a markdown-formatted article.
    It should include headings, intro, body, conclusion, and embed all URLs as markdown links.
    """
    summaries_md = "\n".join(f"- {s}" for s in research_summaries)
    prompt = f"""
You are a professional writer tasked with drafting a comprehensive, polished article in markdown format.

**Topic:** {topic}
**Audience:** {audience}
**Tone:** {tone}

Below are the key research snippets. Use these to inform your writing, and include any URLs you find inside them as clickable markdown links in the text.

Research summaries:
{summaries_md}

**Instructions:**
- Structure the article with a clear introduction, multiple H2/H3 headings for sections, and a concluding summary.
- Embed any URLs from the summaries as [link text](URL) references directly in the narrative.
- At the end of the article, include a "References" section listing each source as a markdown link.

Begin the article now:
"""
    return prompt


@tool
def generate_article(
        topic: str,
        research_summaries: list[str],
        audience: str = "Medium readers",
        tone: str = "informative"
) -> str:
    """
    Generate a polished markdown article based on the provided research summaries.

    Returns:
        A single string containing the full article in markdown, with headings,
        embedded links in the text, and a References section at the end.
    """
    print('***DEBUG***: Running generate_article with topic:', topic)
    article_prompt = _build_article_prompt(topic, research_summaries, audience, tone)
    # Call the LLM and return its output
    response = llm.invoke(article_prompt).content
    return response


def simple_text_summarizer(text: str) -> str:
    # super-naïve fallback: first 500 chars + “…”
    return text[:500].strip() + "…"


@tool
def safe_fetch_and_summarize(url: str) -> str:
    """
    Try the normal summarizer; on error, try a simpler summarizer;
    if that also errors, return a failure message.
    """

    try:
        return fetch_and_summarize(url)
    except Exception as e1:
        print(f"[Warning] fetch_and_summarize failed for {url}: {e1}")
        try:
            # suppose you have a low-level fetch_html tool:
            from langgraph_project.tools.tools import fetch_html
            raw = fetch_html(url)
            return simple_text_summarizer(raw)
        except Exception as e2:
            print(f"[Error] fallback summarizer also failed for {url}: {e2}")
            return f"[Failed to fetch or summarize {url}: {e2}]"

#
