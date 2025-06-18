# preprocess.py

import os
import json
import sqlite3
import re
from bs4 import BeautifulSoup
import html2text
from tqdm import tqdm
import aiohttp
import asyncio
import argparse
import markdown
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory and database paths
DISCOURSE_DIR = "discourse_json"  # Directory containing Discourse JSON exports
MARKDOWN_DIR = "markdown_files"   # Directory containing markdown documents
DB_PATH = "knowledge_base.db"     # SQLite database file

# Ensure required directories exist
os.makedirs(DISCOURSE_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

# Default chunking parameters for splitting text
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieve API key for embedding service from environment
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY environment variable not set. Please set it before running.")

def create_connection():
    """
    Establish a connection to the SQLite database.
    Returns:
        conn (sqlite3.Connection): SQLite connection object or None if failed.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        logger.info(f"Connected to SQLite database at {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def create_tables(conn):
    """
    Create necessary tables for storing discourse and markdown chunks.
    Args:
        conn (sqlite3.Connection): SQLite connection object.
    """
    try:
        cursor = conn.cursor()
        # Table for Discourse post chunks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discourse_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                topic_id INTEGER,
                topic_title TEXT,
                post_number INTEGER,
                author TEXT,
                created_at TEXT,
                likes INTEGER,
                chunk_index INTEGER,
                content TEXT,
                url TEXT,
                embedding BLOB
            )
        ''')
        # Table for markdown document chunks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS markdown_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_title TEXT,
                original_url TEXT,
                downloaded_at TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB
            )
        ''')
        conn.commit()
        logger.info("Database tables created successfully")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")

def create_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split a large text into overlapping chunks for downstream processing.
    Args:
        text (str): Input text to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    Returns:
        list[str]: List of text chunks.
    """
    if not text:
        return []
    chunks = []
    # Normalize whitespace and newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    # Split by paragraphs for more natural chunk boundaries
    paragraphs = text.split('\n')
    current_chunk = ""
    for i, para in enumerate(paragraphs):
        if len(para) > chunk_size:
            # Handle very long paragraphs by splitting into sentences
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = ""
            for sentence in sentences:
                if len(sentence) > chunk_size:
                    # Split very long sentences with overlap
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
                        sentence_chunk = ""
                    for j in range(0, len(sentence), chunk_size - chunk_overlap):
                        sentence_part = sentence[j:j + chunk_size]
                        if sentence_part:
                            chunks.append(sentence_part.strip())
                elif len(sentence_chunk) + len(sentence) > chunk_size and sentence_chunk:
                    chunks.append(sentence_chunk.strip())
                    sentence_chunk = sentence
                else:
                    sentence_chunk = sentence_chunk + " " + sentence if sentence_chunk else sentence
            if sentence_chunk:
                chunks.append(sentence_chunk.strip())
        elif len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = current_chunk + " " + para if current_chunk else para
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    # Add overlap between chunks
    if chunks:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            if len(prev_chunk) > chunk_overlap:
                overlap_start = max(0, len(prev_chunk) - chunk_overlap)
                sentence_break = prev_chunk.rfind('. ', overlap_start)
                if sentence_break != -1 and sentence_break > overlap_start:
                    overlap = prev_chunk[sentence_break+2:]
                    if overlap and not current_chunk.startswith(overlap):
                        current_chunk = overlap + " " + current_chunk
            overlapped_chunks.append(current_chunk)
        return overlapped_chunks
    if text:
        return [text]
    return []

def clean_html(html_content):
    """
    Remove HTML tags and scripts/styles from Discourse post content.
    Args:
        html_content (str): HTML content to clean.
    Returns:
        str: Cleaned plain text.
    """
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_discourse_files(conn):
    """
    Parse all Discourse JSON files, clean and chunk posts, and store them in the database.
    Args:
        conn (sqlite3.Connection): SQLite connection object.
    """
    cursor = conn.cursor()
    # Skip processing if data already exists
    cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info(f"Found {count} existing discourse chunks in database, skipping processing")
        return
    discourse_files = [f for f in os.listdir(DISCOURSE_DIR) if f.endswith('.json')]
    logger.info(f"Found {len(discourse_files)} Discourse JSON files to process")
    total_chunks = 0
    for file_name in tqdm(discourse_files, desc="Processing Discourse files"):
        try:
            file_path = os.path.join(DISCOURSE_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            topic_id = data.get('id')
            topic_title = data.get('title', '')
            topic_slug = data.get('slug', '')
            posts = data.get('post_stream', {}).get('posts', [])
            for post in posts:
                post_id = post.get('id')
                post_number = post.get('post_number')
                author = post.get('username', '')
                created_at = post.get('created_at', '')
                likes = post.get('like_count', 0)
                html_content = post.get('cooked', '')
                clean_content = clean_html(html_content)
                if len(clean_content) < 20:
                    continue  # Skip very short content
                url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}/{post_number}"
                chunks = create_chunks(clean_content)
                for i, chunk in enumerate(chunks):
                    cursor.execute('''
                        INSERT INTO discourse_chunks
                        (post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (post_id, topic_id, topic_title, post_number, author, created_at, likes, i, chunk, url, None))
                    total_chunks += 1
            conn.commit()
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
    logger.info(f"Finished processing Discourse files. Created {total_chunks} chunks.")

def process_markdown_files(conn):
    """
    Parse all markdown files, extract metadata, chunk content, and store in the database.
    Args:
        conn (sqlite3.Connection): SQLite connection object.
    """
    cursor = conn.cursor()
    # Skip processing if data already exists
    cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
    count = cursor.fetchone()[0]
    if count > 0:
        logger.info(f"Found {count} existing markdown chunks in database, skipping processing")
        return
    markdown_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
    logger.info(f"Found {len(markdown_files)} Markdown files to process")
    total_chunks = 0
    for file_name in tqdm(markdown_files, desc="Processing Markdown files"):
        try:
            file_path = os.path.join(MARKDOWN_DIR, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            title = ""
            original_url = ""
            downloaded_at = ""
            # Extract YAML frontmatter if present
            frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if frontmatter_match:
                frontmatter = frontmatter_match.group(1)
                title_match = re.search(r'title: "(.*?)"', frontmatter)
                if title_match:
                    title = title_match.group(1)
                url_match = re.search(r'original_url: "(.*?)"', frontmatter)
                if url_match:
                    original_url = url_match.group(1)
                date_match = re.search(r'downloaded_at: "(.*?)"', frontmatter)
                if date_match:
                    downloaded_at = date_match.group(1)
                # Remove frontmatter from content
                content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
            chunks = create_chunks(content)
            for i, chunk in enumerate(chunks):
                cursor.execute('''
                    INSERT INTO markdown_chunks
                    (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?, NULL)
                ''', (title, original_url, downloaded_at, i, chunk))
                total_chunks += 1
            conn.commit()
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
    logger.info(f"Finished processing Markdown files. Created {total_chunks} chunks.")

async def create_embeddings(api_key):
    """
    Generate embeddings for all discourse and markdown chunks without embeddings.
    Embeddings are created using an external API via aiohttp.
    Args:
        api_key (str): API key for the embedding service.
    """
    if not api_key:
        logger.error("API_KEY environment variable not set. Cannot create embeddings.")
        return
    conn = create_connection()
    cursor = conn.cursor()
    # Fetch discourse and markdown chunks needing embeddings
    cursor.execute("SELECT id, content FROM discourse_chunks WHERE embedding IS NULL")
    discourse_chunks = cursor.fetchall()
    logger.info(f"Found {len(discourse_chunks)} discourse chunks to embed")
    cursor.execute("SELECT id, content FROM markdown_chunks WHERE embedding IS NULL")
    markdown_chunks = cursor.fetchall()
    logger.info(f"Found {len(markdown_chunks)} markdown chunks to embed")

    async def handle_long_text(session, text, record_id, is_discourse=True, max_retries=3):
        """
        Handles embedding of long texts by splitting into subchunks if necessary.
        """
        max_chars = 8000  # Model token limit safety margin
        if len(text) <= max_chars:
            return await embed_text(session, text, record_id, is_discourse, max_retries)
        logger.info(f"Text exceeds embedding limit for {record_id}: {len(text)} chars. Creating multiple embeddings.")
        overlap = 200
        subchunks = []
        for i in range(0, len(text), max_chars - overlap):
            end = min(i + max_chars, len(text))
            subchunk = text[i:end]
            if subchunk:
                subchunks.append(subchunk)
        logger.info(f"Split into {len(subchunks)} subchunks for embedding")
        embeddings = []
        for i, subchunk in enumerate(subchunks):
            logger.info(f"Embedding subchunk {i+1}/{len(subchunks)} for {record_id}")
            success = await embed_text(
                session,
                subchunk,
                record_id,
                is_discourse,
                max_retries,
                f"part_{i+1}_of_{len(subchunks)}"
            )
            if not success:
                logger.error(f"Failed to embed subchunk {i+1}/{len(subchunks)} for {record_id}")
        return True

    async def embed_text(session, text, record_id, is_discourse=True, max_retries=3, part_id=None):
        """
        Calls the embedding API and stores the result in the database.
        Retries on failure.
        """
        retries = 0
        while retries < max_retries:
            try:
                url = "https://aipipe.org/openai/v1/embeddings"
                headers = {
                    "Authorization": api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "text-embedding-3-small",
                    "input": text
                }
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result["data"][0]["embedding"]
                        embedding_blob = json.dumps(embedding).encode()
                        if part_id:
                            # For multi-part embeddings, insert new record with part_id
                            if is_discourse:
                                cursor.execute("""
                                    SELECT post_id, topic_id, topic_title, post_number, author, created_at,
                                           likes, chunk_index, content, url FROM discourse_chunks
                                    WHERE id = ?
                                """, (record_id,))
                                original = cursor.fetchone()
                                if original:
                                    cursor.execute("""
                                        INSERT INTO discourse_chunks
                                        (post_id, topic_id, topic_title, post_number, author, created_at,
                                         likes, chunk_index, content, url, embedding)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        original["post_id"],
                                        original["topic_id"],
                                        original["topic_title"],
                                        original["post_number"],
                                        original["author"],
                                        original["created_at"],
                                        original["likes"],
                                        f"{original['chunk_index']}_{part_id}",
                                        text,
                                        original["url"],
                                        embedding_blob
                                    ))
                            else:
                                cursor.execute("""
                                    SELECT doc_title, original_url, downloaded_at, chunk_index FROM markdown_chunks
                                    WHERE id = ?
                                """, (record_id,))
                                original = cursor.fetchone()
                                if original:
                                    cursor.execute("""
                                        INSERT INTO markdown_chunks
                                        (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                                        VALUES (?, ?, ?, ?, ?, ?)
                                    """, (
                                        original["doc_title"],
                                        original["original_url"],
                                        original["downloaded_at"],
                                        f"{original['chunk_index']}_{part_id}",
                                        text,
                                        embedding_blob
                                    ))
                        else:
                            # Update embedding for original chunk
                            if is_discourse:
                                cursor.execute(
                                    "UPDATE discourse_chunks SET embedding = ? WHERE id = ?",
                                    (embedding_blob, record_id)
                                )
                            else:
                                cursor.execute(
                                    "UPDATE markdown_chunks SET embedding = ? WHERE id = ?",
                                    (embedding_blob, record_id)
                                )
                        conn.commit()
                        return True
                    elif response.status == 429:
                        # Handle API rate limiting with exponential backoff
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        logger.error(f"Error embedding text (status {response.status}): {error_text}")
                        return False
            except Exception as e:
                logger.error(f"Exception embedding text: {e}")
                retries += 1
                await asyncio.sleep(3 * retries)
        logger.error(f"Failed to embed text after {max_retries} retries")
        return False

    batch_size = 10  # Number of chunks to process in parallel
    async with aiohttp.ClientSession() as session:
        # Process discourse chunks in batches
        for i in range(0, len(discourse_chunks), batch_size):
            batch = discourse_chunks[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, True) for record_id, text in batch]
            results = await asyncio.gather(*tasks)
            logger.info(f"Embedded discourse batch {i//batch_size + 1}/{(len(discourse_chunks) + batch_size - 1)//batch_size}: {sum(results)}/{len(batch)} successful")
            if i + batch_size < len(discourse_chunks):
                await asyncio.sleep(2)  # Pause to avoid rate limits
        # Process markdown chunks in batches
        for i in range(0, len(markdown_chunks), batch_size):
            batch = markdown_chunks[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, False) for record_id, text in batch]
            results = await asyncio.gather(*tasks)
            logger.info(f"Embedded markdown batch {i//batch_size + 1}/{(len(markdown_chunks) + batch_size - 1)//batch_size}: {sum(results)}/{len(batch)} successful")
            if i + batch_size < len(markdown_chunks):
                await asyncio.sleep(2)
    conn.close()
    logger.info("Finished creating embeddings")

async def main():
    """
    Main entry point for the preprocessing script.
    Handles argument parsing, database setup, file processing, and embedding creation.
    """
    global CHUNK_SIZE, CHUNK_OVERLAP
    parser = argparse.ArgumentParser(description="Preprocess Discourse posts and markdown files for RAG system")
    parser.add_argument("--api-key", help="API key for aipipe proxy (if not provided, will use API_KEY environment variable)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Size of text chunks (default: {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Overlap between chunks (default: {CHUNK_OVERLAP})")
    args = parser.parse_args()
    # Use API key from argument or environment
    api_key = args.api_key or API_KEY
    if not api_key:
        logger.error("API key not provided. Please provide it via --api-key argument or API_KEY environment variable.")
        return
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    logger.info(f"Using chunk size: {CHUNK_SIZE}, chunk overlap: {CHUNK_OVERLAP}")
    # Set up database
    conn = create_connection()
    if conn is None:
        return
    create_tables(conn)
    # Process Discourse and markdown files
    process_discourse_files(conn)
    process_markdown_files(conn)
    # Generate embeddings for all chunks
    await create_embeddings(api_key)
    conn.close()
    logger.info("Preprocessing complete")

if __name__ == "__main__":
    asyncio.run(main())
