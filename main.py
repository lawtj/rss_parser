"""
Medical RSS Feed Processor
Monitors TBL RSS feed, summarizes new articles, stores in PocketBase
"""

import asyncio
import httpx
import feedparser  # type: ignore
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

_is_loaded = load_dotenv()

# Configuration
POCKETBASE_URL = os.getenv("POCKETBASE_URL")
PB_EMAIL = os.getenv("PB_EMAIL")
PB_PASSWORD = os.getenv("PB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
READER_API_URL = os.getenv("READER_API_URL")
READER_BEARER_TOKEN = os.getenv("READER_BEARER_TOKEN")
PUSHBULLET_TOKEN = os.getenv("PUSHBULLET_TOKEN")


# Validate required environment variables
def validate_env_vars():
    """Validate that all required environment variables are set"""
    required_vars = {
        "POCKETBASE_URL": POCKETBASE_URL,
        "PB_EMAIL": PB_EMAIL,
        "PB_PASSWORD": PB_PASSWORD,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "READER_API_URL": READER_API_URL,
        "READER_BEARER_TOKEN": READER_BEARER_TOKEN,
        "PUSHBULLET_TOKEN": PUSHBULLET_TOKEN,
    }

    missing_vars = [name for name, value in required_vars.items() if not value]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print(f".env file loaded: {_is_loaded}")
        print(f"Current working directory: {os.getcwd()}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


validate_env_vars()

# Global client instances
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_pb_auth_token(client: httpx.AsyncClient) -> str:
    """Authenticate with PocketBase and return token"""
    auth_url = f"{POCKETBASE_URL}/api/collections/_superusers/auth-with-password"
    print(f"DEBUG: Attempting to authenticate with URL: {auth_url}")
    print(f"DEBUG: POCKETBASE_URL = '{POCKETBASE_URL}'")
    response = await client.post(
        auth_url,
        json={"identity": PB_EMAIL, "password": PB_PASSWORD},
    )
    response.raise_for_status()
    try:
        return response.json()["token"]
    except Exception as e:
        print(f"ERROR parsing auth response: {e}")
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        raise


async def fetch_existing_urls(client: httpx.AsyncClient, token: str) -> set[str]:
    """Fetch last 50 URLs from PocketBase for comparison"""
    print("DEBUG: Fetching existing URLs from PocketBase")
    response = await client.get(
        f"{POCKETBASE_URL}/api/collections/rss_feeds/records",
        params={"sort": "-created", "perPage": 50},
        headers={"Authorization": token},
    )
    response.raise_for_status()
    try:
        items = response.json()["items"]
        return {item["url"] for item in items}
    except Exception as e:
        print(f"ERROR parsing fetch_existing_urls response: {e}")
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        raise


async def create_rss_record(
    client: httpx.AsyncClient, item: dict, token: str, summary: str | None = None, markdown: str | None = None
) -> dict:
    """Create new RSS record in PocketBase"""
    print(f"DEBUG: Creating RSS record for {item['link']}")
    data = {
        "url": item["link"],
        "feed_url": item["link"],
        "published_at": item["pubDate"],
    }
    if summary:
        data["summary"] = summary
    if markdown:
        data["markdown"] = markdown

    response = await client.post(
        f"{POCKETBASE_URL}/api/collections/rss_feeds/records",
        json=data,
        headers={"Authorization": token},
    )
    response.raise_for_status()
    try:
        return response.json()
    except Exception as e:
        print(f"ERROR parsing create_rss_record response: {e}")
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        raise


async def fetch_rss(url: str) -> list[dict]:
    """Fetch and parse RSS feed"""
    feed = feedparser.parse(url)
    return [
        {
            "link": entry.link,
            "title": entry.title,
            "pubDate": entry.published,
        }
        for entry in feed.entries
    ]


async def scrape_markdown(client: httpx.AsyncClient, url: str) -> str:
    """Scrape article content as markdown"""
    print(f"DEBUG: Scraping markdown for {url}")
    response = await client.get(
        f"{READER_API_URL}/{url}",
        headers={"Authorization": f"Bearer {READER_BEARER_TOKEN}", "X-Respond-With": "markdown"},
    )
    response.raise_for_status()
    # The API returns plain markdown text, not JSON
    return response.text


async def generate_summary(markdown: str) -> str:
    """Generate LLM summary of article"""
    prompt = f"""You are a medical research assistant. Please analyze this medical research article and provide:
1. A brief summary (2-3 sentences)
2. The main clinical question
3. The rationale for the study
4. Key findings
5. Clinical implications

Format your response as a markdown string with these sections, with the headings bolded: 
summary, clinical_question, rationale, key_findings, clinical_implications

Return the response as pure markdown, without code blocks, backticks, or any other formatting.

Here's the article: {markdown}"""

    response = await openai_client.chat.completions.create(
        model="gpt-5-mini", messages=[{"role": "user", "content": prompt}]
    )
    if response.choices[0].message.content is None:
        raise ValueError("LLM returned no content")
    return response.choices[0].message.content


async def send_pushbullet(client: httpx.AsyncClient, title: str, body: str):
    """Send Pushbullet notification"""
    response = await client.post(
        "https://api.pushbullet.com/v2/pushes",
        headers={"Access-Token": PUSHBULLET_TOKEN},
        json={"type": "note", "title": title, "body": body},
    )
    response.raise_for_status()


async def process_item(client: httpx.AsyncClient, item: dict, token: str):
    """Process a single RSS item"""
    print(f"Processing: {item['title']}")

    # Skip evidence-updates
    if "evidence-updates" in item["link"].lower():
        print(f"Skipping evidence-updates: {item['title']}")
        # Create minimal record for evidence-updates
        await create_rss_record(client, item, token)
        print(f"Created record for: {item['title']}")
        return

    # Scrape and summarize BEFORE creating record
    markdown = await scrape_markdown(client, item["link"])
    summary = await generate_summary(markdown)

    # Only create record after all processing succeeds
    await create_rss_record(client, item, token, summary=summary, markdown=markdown)
    print(f"Created record for: {item['title']}")

    # Notify
    await send_pushbullet(client, "New Summary", summary)
    print(f"Completed: {item['title']}")


async def main():
    """Main execution"""
    print(f"Starting RSS processing at {datetime.now()}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Authenticate
        token = await get_pb_auth_token(client)

        # Fetch existing URLs from PocketBase
        existing_urls = await fetch_existing_urls(client, token)
        print(f"Found {len(existing_urls)} existing URLs in PocketBase")

        # Fetch RSS
        items = await fetch_rss("https://www.thebottomline.org.uk/feed/")
        print(f"Found {len(items)} RSS items")

        # Filter to only new items
        new_items = [item for item in items if item["link"] not in existing_urls]
        print(f"Found {len(new_items)} new items to process")

        # Process items
        for item in new_items:
            try:
                await process_item(client, item, token)
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error processing {item['title']}: {e}")

        print("Processing complete")


if __name__ == "__main__":
    asyncio.run(main())
