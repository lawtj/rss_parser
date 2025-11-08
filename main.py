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
    return response.json()["token"]


async def check_url_exists(client: httpx.AsyncClient, url: str, token: str) -> dict | None:
    """Check if URL already exists in PocketBase"""
    response = await client.get(
        f"{POCKETBASE_URL}/api/collections/rss_feeds/records",
        params={"filter": f'url="{url}"', "perPage": 1},
        headers={"Authorization": token},
    )
    response.raise_for_status()
    items = response.json()["items"]
    return items[0] if items else None


async def create_rss_record(client: httpx.AsyncClient, item: dict, token: str) -> dict:
    """Create new RSS record in PocketBase"""
    response = await client.post(
        f"{POCKETBASE_URL}/api/collections/rss_feeds/records",
        json={"url": item["link"], "feed_url": item["link"], "published_at": item["pubDate"]},
        headers={"Authorization": token},
    )
    response.raise_for_status()
    return response.json()


async def update_rss_record(client: httpx.AsyncClient, record_id: str, summary: str, markdown: str, token: str):
    """Update record with summary and markdown"""
    response = await client.patch(
        f"{POCKETBASE_URL}/api/collections/rss_feeds/records/{record_id}",
        json={"summary": summary, "markdown": markdown},
        headers={"Authorization": token},
    )
    response.raise_for_status()


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
    response = await client.get(
        f"{READER_API_URL}/{url}",
        headers={"Authorization": f"Bearer {READER_BEARER_TOKEN}", "X-Respond-With": "markdown"},
    )
    response.raise_for_status()
    return response.json()["data"]


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
    # Check if exists
    existing = await check_url_exists(client, item["link"], token)
    if existing:
        print(f"Skipping existing: {item['title']}")
        return

    # Create record
    record = await create_rss_record(client, item, token)
    print(f"Created record for: {item['title']}")

    # Only process evidence-updates
    if "evidence-updates" in item["link"].lower():
        print(f"Skipping evidence-updates: {item['title']}")
        return

    # Scrape and summarize
    print(f"Processing: {item['title']}")
    markdown = await scrape_markdown(client, item["link"])
    summary = await generate_summary(markdown)

    # Update record
    await update_rss_record(client, record["id"], summary, markdown, token)

    # Notify
    await send_pushbullet(client, "New Summary", summary)
    print(f"Completed: {item['title']}")


async def main():
    """Main execution"""
    print(f"Starting RSS processing at {datetime.now()}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Authenticate
        token = await get_pb_auth_token(client)

        # Fetch RSS
        items = await fetch_rss("https://www.thebottomline.org.uk/feed/")
        print(f"Found {len(items)} RSS items")

        # Process items
        for item in items:
            try:
                await process_item(client, item, token)
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error processing {item['title']}: {e}")

        print("Processing complete")


if __name__ == "__main__":
    asyncio.run(main())
