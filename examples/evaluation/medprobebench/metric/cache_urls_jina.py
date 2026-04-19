# -*- coding: utf-8 -*-
"""URL content caching module - Jina Reader version."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import threading
import random  
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.parse

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


JINA_READER_BASE_URL = "https://r.jina.ai/"
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")


@dataclass
class URLCacheEntry:
    """URL cache entry."""
    url: str
    content: str
    fetch_time: float
    status_code: int = 200
    error: str = ""
    content_type: str = ""
    source: str = "jina"
    
    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "content": self.content,
            "fetch_time": self.fetch_time,
            "status_code": self.status_code,
            "error": self.error,
            "content_type": self.content_type,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "URLCacheEntry":
        return cls(
            url=d.get("url", ""),
            content=d.get("content", ""),
            fetch_time=d.get("fetch_time", 0.0),
            status_code=d.get("status_code", 200),
            error=d.get("error", ""),
            content_type=d.get("content_type", ""),
            source=d.get("source", "jina"),
        )


def _is_cloudflare_block(content: str) -> bool:
    if not content or len(content) >= 500:
        return False
    
    content_lower = content.lower()
    indicators = ["verify you are human", "cloudflare", "ray id:", "checking your browser"]
    matches = sum(1 for ind in indicators if ind in content_lower)
    return matches >= 2


def _is_access_denied(content: str) -> bool:
    if not content or len(content) >= 300:
        return False
    
    content_lower = content.lower()
    indicators = ["access denied", "403 forbidden", "404 not found", "page not found"]
    return any(ind in content_lower for ind in indicators)

def _extract_doi_from_url(url: str) -> str | None: 
    if not url: 
        return None 
    patterns = [ r'doi\.org/(10\.\d{4,}/[^\s&?#]+)', r'(10\.\d{4,}/[^\s&?#]+)', ] 
    for pattern in patterns: 
        match = re.search(pattern, url, re.IGNORECASE) 
        if match: return match.group(1).rstrip('.,;:!?)') 
    return None 

def _extract_pmid_from_url(url: str) -> str | None: 
    if not url: 
        return None
    patterns = [ r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', r'ncbi\.nlm\.nih\.gov/pubmed/(\d+)', r'pmid[=:/](\d+)', ] 
    for pattern in patterns: 
        match = re.search(pattern, url, re.IGNORECASE) 
        if match: 
            return match.group(1) 
    return None

def _extract_pmcid_from_url(url: str) -> str | None:
    """
    Support format:
    - https://pmc.ncbi.nlm.nih.gov/articles/PMC12691433/
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12691433/
    - PMC12691433
    """
    if not url:
        return None
    
    patterns = [
        r'pmc\.ncbi\.nlm\.nih\.gov/articles/PMC(\d+)',
        r'ncbi\.nlm\.nih\.gov/pmc/articles/PMC(\d+)',
        r'/PMC(\d+)',
        r'PMC(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)  # Return numeric part without "PMC" prefix
    return None


def _extract_readable_text(content: str, source: str = "unknown") -> str:
    """Extract readable text from XML/JSON content
    """
    if not content:
        return ""
    
    content_stripped = content.strip()
    
    # Detect and process XML
    if content_stripped.startswith('<'):
        return _extract_from_xml(content)
    
    # Detect and process JSON
    if content_stripped.startswith('{'):
        return _extract_from_json(content)
    
    # HTML
    if '<html' in content_stripped[:500].lower():
        return _extract_from_html(content)
    
    # Already plain text, return directly
    return content

def _extract_from_html(html: str) -> str:
    text = re.sub(r'<script.*?>.*?</script>', ' ', html, flags=re.DOTALL)
    text = re.sub(r'<style.*?>.*?</style>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text[:12000]

def _extract_from_xml(content: str, max_chars: int = 12000) -> str:
    if not content:
        return ""

    if len(content) > 2_000_000:
        content = content[:2_000_000]

    extracted_parts = []

    title_patterns = [
        r'<ArticleTitle[^>]*>(.*?)</ArticleTitle>',
        r'<article-title[^>]*>(.*?)</article-title>',
    ]
    for pat in title_patterns:
        m = re.search(pat, content, re.IGNORECASE | re.DOTALL)
        if m:
            title = re.sub(r'<[^>]+>', '', m.group(1)).strip()
            if title:
                extracted_parts.append(f"Title: {title}")
                break

    abstract_patterns = [
        r'<Abstract[^>]*>(.*?)</Abstract>',
        r'<abstract[^>]*>(.*?)</abstract>',
    ]
    for pat in abstract_patterns:
        m = re.search(pat, content, re.IGNORECASE | re.DOTALL)
        if m:
            abstract_block = m.group(1)
            paras = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', abstract_block, re.IGNORECASE | re.DOTALL)
            if not paras:
                paras = [abstract_block]
            texts = []
            for p in paras:
                t = re.sub(r'<[^>]+>', ' ', p)
                t = re.sub(r'\s+', ' ', t).strip()
                if len(t) > 30:
                    texts.append(t)
            if texts:
                extracted_parts.append("Abstract: " + " ".join(texts))
                break

    body_texts = []

    # Prioritize extraction of <body> (PMC / JATS)
    body_match = re.search(
        r'<body[^>]*>(.*?)</body>',
        content,
        re.IGNORECASE | re.DOTALL
    )

    body_content = body_match.group(1) if body_match else content

    # extraction of section
    paragraphs = re.findall(
        r'<p[^>]*>(.*?)</p>',
        body_content,
        re.IGNORECASE | re.DOTALL
    )

    for p in paragraphs:
        text = re.sub(r'<[^>]+>', ' ', p)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 50:   
            body_texts.append(text)

        if sum(len(t) for t in body_texts) > max_chars:
            break

    if body_texts:
        extracted_parts.append("Body:\n" + "\n".join(body_texts))

    author_match = re.search(
        r'<contrib-group[^>]*>(.*?)</contrib-group>',
        content,
        re.IGNORECASE | re.DOTALL
    )
    if author_match:
        names = re.findall(
            r'<surname[^>]*>(.*?)</surname>',
            author_match.group(1),
            re.IGNORECASE
        )
        if names:
            extracted_parts.append("Authors: " + ", ".join(names[:5]))

    year_match = re.search(
        r'<year[^>]*>(\d{4})</year>',
        content,
        re.IGNORECASE
    )
    if year_match:
        extracted_parts.append(f"Year: {year_match.group(1)}")

    if extracted_parts:
        return "\n\n".join(extracted_parts)

    text = re.sub(r'<[^>]+>', ' ', content)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]

def _extract_from_json(content: str) -> str:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content[:3000] 
    
    extracted_parts = []
    
    if "message" in data:
        msg = data["message"]
        
        titles = msg.get("title", [])
        if titles:
            extracted_parts.append(f"Title: {titles[0]}")
        
        authors = msg.get("author", [])
        if authors:
            names = [f"{a.get('family', '')} {a.get('given', '')}".strip() for a in authors[:5]]
            extracted_parts.append(f"Authors: {', '.join(names)}")
        
        container = msg.get("container-title", [])
        if container:
            extracted_parts.append(f"Journal: {container[0]}")
        
        published = msg.get("published", {}).get("date-parts", [[]])
        if published and published[0]:
            extracted_parts.append(f"Year: {published[0][0]}")
        
        doi = msg.get("DOI", "")
        if doi:
            extracted_parts.append(f"DOI: {doi}")
        
        abstract = msg.get("abstract", "")
        if abstract:
            abstract = re.sub(r'<[^>]+>', '', abstract).strip()
            extracted_parts.append(f"Abstract: {abstract}")
    
    else:
        priority_keys = ['title', 'abstract', 'summary', 'description', 'author', 'year']
        found = _find_values_recursive(data, priority_keys)
        for key, value in found.items():
            extracted_parts.append(f"{key.capitalize()}: {value}")
    
    if extracted_parts:
        return "\n".join(extracted_parts)
    
    return json.dumps(data, ensure_ascii=False)[:3000]

def _find_values_recursive(obj: Any, keys: list[str], found: dict | None = None, depth: int = 0) -> dict:
    if found is None:
        found = {}
    
    if depth > 5:
        return found
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = key.lower()
            for pk in keys:
                if pk in key_lower and pk not in found:
                    if isinstance(value, str) and len(value) > 10:
                        found[pk] = value[:500]
                    elif isinstance(value, list) and value and isinstance(value[0], str):
                        found[pk] = value[0][:500]
            _find_values_recursive(value, keys, found, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:3]:
            _find_values_recursive(item, keys, found, depth + 1)
    
    return found


def normalize_pmcid(pmcid: str | None) -> str | None:
    if not pmcid:
        return None
    return pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"

async def _try_alternative_sources_async(url: str, timeout: int = 30) -> tuple[str, int, str]:
    """Attempt to obtain the content from backup data sources.
    
    Optimized sequence (sorted by success rate and content quality):
    1. Direct acquisition from PMC (if PMCID exists)
    2. PubMed EFetch - Highest success rate, with abstract
    3. Semantic Scholar - Broad coverage, with abstract
    4. CrossRef - The most extensive coverage, with metadata
    5. OpenAlex - Emerging open data source
    6. Europe PMC - Possibly full text
    7. PMC OAI - Open access full text (via PMID conversion)
    8. Unpaywall - Search for open access versions
    """
    if not HAS_HTTPX:
        return "", 0, "need httpx"

    doi = _extract_doi_from_url(url)
    pmid = _extract_pmid_from_url(url)
    pmcid = _extract_pmcid_from_url(url) 

    if not doi and not pmid and not pmcid:
        return "", 0, "Unable to extract DOI or PMID or PMCID"
    
    
    headers = {
        "User-Agent": "GuideBench/1.0 (Academic Research; mailto:research@example.com)",
        "Accept": "application/json, application/xml, text/plain",
    }

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        
        if doi and not pmid:
            try:
                idconv_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={doi}&format=json"
                resp = await client.get(idconv_url)
                if resp.status_code == 200:
                    records = resp.json().get("records", [])
                    if records:
                        pmid = records[0].get("pmid")
                        pmcid = records[0].get("pmcid")
                        if pmcid:
                            pmcid = normalize_pmcid(pmcid)
            except Exception:
                pass
            
            if not pmid:
                try:
                    esearch_url = (
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                        f"?db=pubmed&term={doi}[doi]&retmode=json"
                    )
                    resp = await client.get(esearch_url)
                    if resp.status_code == 200:
                        data = resp.json()
                        id_list = data.get("esearchresult", {}).get("idlist", [])
                        if id_list:
                            pmid = id_list[0]
                except Exception:
                    pass
        
        if pmcid:
            pmcid = normalize_pmcid(pmcid)
            try:
                oai_url = (
                    "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
                    f"?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid.replace('PMC','')}&metadataPrefix=pmc"
                )
                resp = await client.get(oai_url)
                if resp.status_code == 200 and len(resp.text) > 500:
                    readable = _extract_readable_text(resp.text, "pmc_oai")
                    if readable and len(readable) > 100:
                        return readable, 200, ""
            except Exception:
                pass
            
            try:
                fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                resp = await client.get(fulltext_url)
                if resp.status_code == 200 and len(resp.text) > 500:
                    readable = _extract_readable_text(resp.text, "europepmc")
                    if readable and len(readable) > 100:
                        return readable, 200, ""
            except Exception:
                pass
            
            try:
                idconv_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmcid}&format=json"
                resp = await client.get(idconv_url)
                if resp.status_code == 200:
                    records = resp.json().get("records", [])
                    if records and "pmid" in records[0]:
                        converted_pmid = records[0]["pmid"]
                        efetch_url = (
                            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                            f"?db=pubmed&id={converted_pmid}&retmode=xml"
                        )
                        efetch_resp = await client.get(efetch_url)
                        if efetch_resp.status_code == 200 and len(efetch_resp.text) > 200:
                            readable = _extract_readable_text(efetch_resp.text, "pubmed_efetch")
                            if readable and len(readable) > 100:
                                return readable, 200, ""
            except Exception:
                pass

        if pmid:
            try:
                efetch_url = (
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    f"?db=pubmed&id={pmid}&retmode=xml"
                )
                resp = await client.get(efetch_url)
                if resp.status_code == 200 and len(resp.text) > 200:
                    readable = _extract_readable_text(resp.text, "pubmed_efetch")
                    if readable and len(readable) > 100:
                        return readable, 200, ""
            except Exception:
                pass

        if doi or pmid or pmcid:
            try:
                if doi:
                    ss_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract,authors,year,venue,citationCount"
                elif pmid:
                    ss_url = f"https://api.semanticscholar.org/graph/v1/paper/PMID:{pmid}?fields=title,abstract,authors,year,venue,citationCount"
                elif pmcid:
                    ss_url = f"https://api.semanticscholar.org/graph/v1/paper/PMCID:{pmcid}?fields=title,abstract,authors,year,venue,citationCount"
                
                resp = await client.get(ss_url)
                if resp.status_code == 200:
                    data = resp.json()
                    readable = _extract_semantic_scholar_content(data)
                    if readable and len(readable) > 100:
                        return readable, 200, ""

                if resp.status_code == 429:
                    await asyncio.sleep(1.5)
                
            except Exception:
                pass

        if doi:
            try:
                crossref_url = f"https://api.crossref.org/works/{doi}"
                resp = await client.get(crossref_url)
                if resp.status_code == 200:
                    readable = _extract_readable_text(resp.text, "crossref")
                    if readable and len(readable) > 100:
                        return readable, 200, ""
            except Exception:
                pass

        if doi:
            try:
                encoded_doi = urllib.parse.quote(f"https://doi.org/{doi}", safe='')
                openalex_url = f"https://api.openalex.org/works/{encoded_doi}"
                
                resp = await client.get(openalex_url)
                if resp.status_code == 200:
                    data = resp.json()
                    readable = _extract_openalex_content(data)
                    if readable and len(readable) > 100:
                        return readable, 200, ""
            except Exception:
                pass

        if doi or pmid or pmcid:
            try:
                if pmcid:
                    identifier = f"{pmcid}"
                else:
                    identifier = doi if doi else pmid
                identifier = urllib.parse.quote(identifier)
                search_url = (
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                    f"?query={identifier}&format=json&resultType=core"
                )
                resp = await client.get(search_url)
                if resp.status_code == 200:
                    results = resp.json().get("resultList", {}).get("result", [])
                    if results:
                        result_pmcid = results[0].get("pmcid")
                        if result_pmcid:
                            fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{result_pmcid}/fullTextXML"
                            full_resp = await client.get(fulltext_url)
                            if full_resp.status_code == 200 and len(full_resp.text) > 500:
                                readable = _extract_readable_text(full_resp.text, "europepmc")
                                if readable and len(readable) > 100:
                                    return readable, 200, ""
                        
                        readable = _extract_europepmc_result(results[0])
                        if readable and len(readable) > 100:
                            return readable, 200, ""
            except Exception:
                pass

        if pmid and not pmcid:  
            try:
                idconv_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
                resp = await client.get(idconv_url)
                if resp.status_code == 200:
                    records = resp.json().get("records", [])
                    if records and "pmcid" in records[0]:
                        converted_pmcid = records[0]["pmcid"].replace("PMC", "")
                        oai_url = (
                            "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
                            f"?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{converted_pmcid}&metadataPrefix=pmc"
                        )
                        full = await client.get(oai_url)
                        if full.status_code == 200 and len(full.text) > 500:
                            readable = _extract_readable_text(full.text, "pmc_oai")
                            if readable and len(readable) > 100:
                                return readable, 200, ""
            except Exception:
                pass

        if doi:
            try:
                unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=research@example.com"
                resp = await client.get(unpaywall_url)
                if resp.status_code == 200:
                    data = resp.json()
                    readable = _extract_unpaywall_content(data)
                    if readable and len(readable) > 100:
                        return readable, 200, ""
            except Exception:
                pass

        return "", 0, "All backup sources have failed."

def _extract_semantic_scholar_content(data: dict) -> str:
    parts = []
    
    if data.get("title"):
        parts.append(f"Title: {data['title']}")
    
    if data.get("authors"):
        author_names = [a.get("name", "") for a in data["authors"][:5]]
        parts.append(f"Authors: {', '.join(author_names)}")
    
    if data.get("venue"):
        parts.append(f"Venue: {data['venue']}")
    
    if data.get("year"):
        parts.append(f"Year: {data['year']}")
    
    if data.get("citationCount"):
        parts.append(f"Citations: {data['citationCount']}")
    
    if data.get("abstract"):
        parts.append(f"Abstract: {data['abstract']}")
    
    return "\n".join(parts)

def _extract_openalex_content(data: dict) -> str:
    parts = []
    
    if data.get("title"):
        parts.append(f"Title: {data['title']}")
    
    if data.get("authorships"):
        author_names = [
            a.get("author", {}).get("display_name", "") 
            for a in data["authorships"][:5]
        ]
        parts.append(f"Authors: {', '.join(filter(None, author_names))}")
    
    if data.get("primary_location", {}).get("source", {}).get("display_name"):
        parts.append(f"Journal: {data['primary_location']['source']['display_name']}")
    
    if data.get("publication_year"):
        parts.append(f"Year: {data['publication_year']}")
    
    if data.get("cited_by_count"):
        parts.append(f"Citations: {data['cited_by_count']}")
    
    if data.get("abstract_inverted_index"):
        try:
            inverted = data["abstract_inverted_index"]
            words = [(word, min(positions)) for word, positions in inverted.items()]
            words.sort(key=lambda x: x[1])
            abstract = " ".join(word for word, _ in words)
            parts.append(f"Abstract: {abstract}")
        except Exception:
            pass
    
    return "\n".join(parts)

def _extract_europepmc_result(result: dict) -> str:
    parts = []
    
    if result.get("title"):
        parts.append(f"Title: {result['title']}")
    
    if result.get("authorString"):
        parts.append(f"Authors: {result['authorString']}")
    
    if result.get("journalTitle"):
        parts.append(f"Journal: {result['journalTitle']}")
    
    if result.get("pubYear"):
        parts.append(f"Year: {result['pubYear']}")
    
    if result.get("doi"):
        parts.append(f"DOI: {result['doi']}")
    
    if result.get("abstractText"):
        parts.append(f"Abstract: {result['abstractText']}")
    
    return "\n".join(parts)

def _extract_unpaywall_content(data: dict) -> str:
    parts = []
    
    if data.get("title"):
        parts.append(f"Title: {data['title']}")
    
    if data.get("z_authors"):
        author_names = [
            f"{a.get('family', '')} {a.get('given', '')}".strip()
            for a in data["z_authors"][:5]
        ]
        parts.append(f"Authors: {', '.join(filter(None, author_names))}")
    
    if data.get("journal_name"):
        parts.append(f"Journal: {data['journal_name']}")
    
    if data.get("year"):
        parts.append(f"Year: {data['year']}")
    
    if data.get("doi"):
        parts.append(f"DOI: {data['doi']}")
    
    if data.get("is_oa"):
        parts.append(f"Open Access: Yes")
        if data.get("best_oa_location", {}).get("license"):
            parts.append(f"License: {data['best_oa_location']['license']}")
    
    return "\n".join(parts)

class JinaReaderClient:
    
    def __init__(self, api_key: str | None = None, timeout: int = 60):  
        self.api_key = api_key or JINA_API_KEY
        self.timeout = timeout
        self.base_url = JINA_READER_BASE_URL
        
        self._headers = {
            "Accept": "text/plain",
            "X-Return-Format": "text",
            "X-With-Generated-Alt": "true",
            "X-No-Cache": "true", 
        }
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"
        
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
    
    
    def _get_async_client(self) -> "httpx.AsyncClient":
        if not HAS_HTTPX:
            raise ImportError("need install httpx: pip install httpx")
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers=self._headers,
            )
        return self._async_client
    
    
    async def read_async(self, url: str) -> tuple[str, int, str]:
        jina_url = f"{self.base_url}{url}"
        
        try:
            client = self._get_async_client()
            response = await client.get(jina_url)
            response.raise_for_status()
            content = response.text.strip()
            
            if not content or len(content) < 50:
                return "", response.status_code, "Content is empty or too short."
            
            return content, response.status_code, ""
            
        except httpx.HTTPStatusError as e:
            return "", e.response.status_code, f"HTTP {e.response.status_code}"
            
        except httpx.TimeoutException:
            return "", 0, "Request timeout"
        
        except httpx.ConnectError:
            return "", 0, "Connection failed"
        
        except httpx.ReadTimeout:
            return "", 0, "Read timeout"
            
        except Exception as e:
            return "", 0, str(e)[:100]
    
    
    async def aclose(self):
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None


class URLCacheManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        fetch_timeout: int = 10,  
        max_cache_length: int = 15000,
        max_workers: int = 10,  
        max_retries: int = 3,   
        enable_disk_cache: bool = True,
        verbose: bool = False,
        jina_api_key: str | None = None,
        request_delay: float = 0.5,  
    ):
        if self._initialized:
            return
        
        self._initialized = True
        self.fetch_timeout = fetch_timeout
        self.max_cache_length = max_cache_length
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.enable_disk_cache = enable_disk_cache
        self.verbose = verbose
        
        self._jina_client = JinaReaderClient(api_key=jina_api_key, timeout=fetch_timeout)
        self._memory_cache: dict[str, URLCacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None
        self.request_delay = request_delay
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".url_cache_jina"
        self.cache_dir = Path(cache_dir)
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "fetch_count": 0,
            "fetch_success": 0,
            "fetch_failed": 0,
            "total_fetch_time": 0.0,
            "retry_count": 0,
        }

    def _get_async_lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock
    
    def _url_to_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_file_path(self, url: str) -> Path:
        return self.cache_dir / f"{self._url_to_cache_key(url)}.json"
    
    def _load_from_disk(self, url: str) -> URLCacheEntry | None:
        if not self.enable_disk_cache:
            return None
        cache_file = self._get_cache_file_path(url)
        if not cache_file.exists():
            return None
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                entry = URLCacheEntry.from_dict(json.load(f))
                with self._cache_lock:
                    self._memory_cache[url] = entry
                return entry
        except Exception:
            return None
    
    def _save_to_disk(self, entry: URLCacheEntry):
        if not self.enable_disk_cache:
            return
        try:
            with self._get_cache_file_path(entry.url).open("w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _remove_from_disk(self, url: str):
        if self.enable_disk_cache:
            try:
                cache_file = self._get_cache_file_path(url)
                if cache_file.exists():
                    cache_file.unlink()
            except Exception:
                pass

    def _is_valid_cache(self, entry: URLCacheEntry | None) -> bool:
        if entry is None or not entry.content or len(entry.content.strip()) < 50:
            return False
        if _is_cloudflare_block(entry.content) or _is_access_denied(entry.content):
            return False
        if entry.error and any(e in entry.error for e in ["Cloudflare", "access denied", "verification wall"]):
            return False
        return True

    
    def _should_retry(self, status_code: int, error: str, content: str) -> tuple[bool, float]:
        if _is_cloudflare_block(content):
            return True, 8.0 + random.uniform(0, 2) 
        
        if status_code == 429:
            return True, 15.0 + random.uniform(0, 5)  # 15-20 seconds
        
        if 500 <= status_code < 600:
            return True, 5.0 + random.uniform(0, 2)
        
        if "timeout" in error.lower() or "connect" in error.lower():
            return True, 5.0 + random.uniform(0, 2)
        
        if status_code in (401, 403):
            return True, 3.0 + random.uniform(0, 1)
        
        if _is_access_denied(content):
            return True, 5.0 + random.uniform(0, 2)
        
        if not content or len(content.strip()) < 100:
            return True, 3.0 + random.uniform(0, 1)
        
        if 400 <= status_code < 500:
            return True, 3.0 + random.uniform(0, 1)
        
        if error:
            return True, 5.0 + random.uniform(0, 2)
        
        return False, 0.0

    async def _fetch_and_cache_async(self, url: str) -> str:
        self.stats["fetch_count"] += 1
        start_time = time.perf_counter()
        
        content = ""
        status_code = 0
        error = ""
        tried_alternative = False
        
        pmid = _extract_pmid_from_url(url)
        pmcid = _extract_pmcid_from_url(url)
        is_pubmed_url = pmid or pmcid or "pubmed" in url.lower() or "pmc.ncbi" in url.lower()
        
        if is_pubmed_url and not tried_alternative:
            tried_alternative = True
            if self.verbose:
                print(f"      🔬 Detected PubMed/PMC URL, prioritizing alternative sources: {url[:50]}...")
            
            alt_content, alt_status, alt_error = await _try_alternative_sources_async(url, self.fetch_timeout)
            if alt_content and len(alt_content.strip()) > 100:
                content = alt_content[:self.max_cache_length] if len(alt_content) > self.max_cache_length else alt_content
                status_code = alt_status
                error = ""
                
                elapsed = time.perf_counter() - start_time
                self.stats["total_fetch_time"] += elapsed
                self.stats["fetch_success"] += 1
                
                if self.verbose:
                    print(f"      ✅ Alternative source succeeded: {url[:50]}... ({len(content)} chars, {elapsed:.1f}s)")
                
                entry = URLCacheEntry(
                    url=url,
                    content=content,
                    fetch_time=time.time(),
                    status_code=status_code,
                    error="",
                    source="alternative",
                )
                
                async_lock = self._get_async_lock()
                async with async_lock:
                    self._memory_cache[url] = entry
                self._save_to_disk(entry)
                
                return content
            
            if self.verbose:
                print(f"      ⚠️ Alternative source failed: {alt_error}, trying Jina Reader...")
        
        for attempt in range(self.max_retries):
            if attempt > 0 or self.request_delay > 0:
                delay = self.request_delay + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
            
            content, status_code, error = await self._jina_client.read_async(url)
            
            if self.verbose:
                print(f"      📡 Attempt {attempt + 1}: {url[:60]}... -> status={status_code}, len={len(content) if content else 0}, error={error[:50] if error else 'None'}")
            
            if content and len(content) > self.max_cache_length:
                content = content[:self.max_cache_length]
            
            is_valid = (
                content and 
                len(content.strip()) > 100 and 
                not _is_cloudflare_block(content) and 
                not _is_access_denied(content)
            )
            
            if is_valid:
                if self.verbose and attempt > 0:
                    print(f"      ✅ Successfully fetched (attempt {attempt + 1}): {url[:50]}...")
                break
            
            should_retry, wait_time = self._should_retry(status_code, error, content)
            
            should_try_alternative = (
                not tried_alternative and (
                    status_code in (401, 403, 0) or  
                    _is_access_denied(content) or
                    "timeout" in error.lower() or
                    "connect" in error.lower()
                )
            )
            
            if should_try_alternative:
                tried_alternative = True
                if self.verbose:
                    print(f"      ℹ️ Jina failed (status={status_code}, error={error[:30]}), trying alternative sources: {url[:50]}...")
                
                await asyncio.sleep(1.0 + random.uniform(0, 0.5))
                
                alt_content, alt_status, alt_error = await _try_alternative_sources_async(url, self.fetch_timeout)
                if alt_content and len(alt_content.strip()) > 100:
                    content = alt_content[:self.max_cache_length] if len(alt_content) > self.max_cache_length else alt_content
                    status_code = alt_status
                    error = ""
                    if self.verbose:
                        print(f"      ✅ Alternative source succeeded: {url[:50]}... ({len(content)} chars)")
                    break
                if self.verbose:
                    print(f"      ⚠️ Alternative source also failed: {alt_error}, continuing retries on main source...")
            
            if attempt >= self.max_retries - 1:
                if self.verbose:
                    print(f"      ❌ All retries exhausted ({self.max_retries} attempts): {url[:50]}...")
                break
            
            if should_retry:
                self.stats["retry_count"] += 1
                actual_wait = wait_time * (1.5 ** attempt) + random.uniform(0, 1)
                actual_wait = min(actual_wait, 60.0)  
                
                if self.verbose:
                    reason = error if error else f"HTTP {status_code}" if status_code else "invalid content"
                    print(f"      ⏳ Retry {attempt + 1}/{self.max_retries}: waiting {actual_wait:.1f}s ({reason})")
                
                await asyncio.sleep(actual_wait)
            else:
                if self.verbose:
                    print(f"      ℹ️ No retry needed, exiting: {url[:50]}...")
                break
        
        # ...existing post-processing...
        elapsed = time.perf_counter() - start_time
        self.stats["total_fetch_time"] += elapsed
        
        is_valid = (
            content and 
            len(content.strip()) > 100 and 
            not _is_cloudflare_block(content) and 
            not _is_access_denied(content)
        )
        
        if is_valid:
            self.stats["fetch_success"] += 1
            if self.verbose:
                print(f"      ✓ Final fetch succeeded: {url[:50]}... ({len(content)} chars, {elapsed:.1f}s)")
            
            entry = URLCacheEntry(
                url=url,
                content=content,
                fetch_time=time.time(),
                status_code=status_code,
                error="",
                source="jina",
            )
            
            async_lock = self._get_async_lock()
            async with async_lock:
                self._memory_cache[url] = entry
            self._save_to_disk(entry)
            
            return content
        else:
            self.stats["fetch_failed"] += 1
            if self.verbose:
                reason = error if error else f"HTTP {status_code}" if status_code else "content invalid or too short"
                print(f"      ✗ Final fetch failed: {url[:50]}... ({reason}, {elapsed:.1f}s)")
            return ""

    async def get_batch_async(self, urls: list[str]) -> dict[str, str]:
        """Asynchronous batch fetch."""
        results = {}
        unique_urls = list(set(url.strip() for url in urls if url.strip()))
        urls_to_fetch = []
        
        for url in unique_urls:
            if url in self._memory_cache:
                entry = self._memory_cache[url]
                if self._is_valid_cache(entry):
                    results[url] = entry.content
                    self.stats["memory_hits"] += 1
                    continue
                with self._cache_lock:
                    if url in self._memory_cache:
                        del self._memory_cache[url]
            
            entry = self._load_from_disk(url)
            if self._is_valid_cache(entry):
                results[url] = entry.content
                self.stats["disk_hits"] += 1
                continue
            elif entry is not None:
                self._remove_from_disk(url)
            
            urls_to_fetch.append(url)
        
        if not urls_to_fetch:
            return results
        
        if self.verbose:
            print(f"    📥 Need to fetch {len(urls_to_fetch)} URLs (cache hits: {len(unique_urls) - len(urls_to_fetch)})")
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def fetch_with_semaphore(url: str, index: int) -> tuple[str, str]:
            async with semaphore:
                initial_delay = (index % self.max_workers) * 0.2 + random.uniform(0, 0.3)
                await asyncio.sleep(initial_delay)
                return url, await self._fetch_and_cache_async(url)
        
        tasks = [fetch_with_semaphore(url, i) for i, url in enumerate(urls_to_fetch)]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        fail_count = 0
        
        for result in fetch_results:
            if isinstance(result, Exception):
                fail_count += 1
                if self.verbose:
                    print(f"      ⚠️ Task exception: {result}")
                continue
            url, content = result
            results[url] = content
            if content:
                success_count += 1
            else:
                fail_count += 1
        
        if self.verbose:
            print(f"    📊 Batch fetch completed: success {success_count}, failed {fail_count}, total {len(urls_to_fetch)}")
        
        return results

    async def get_async(self, url: str) -> str:
        url = url.strip()
        if not url:
            return ""
        
        if url in self._memory_cache:
            entry = self._memory_cache[url]
            if self._is_valid_cache(entry):
                self.stats["memory_hits"] += 1
                return entry.content
            async_lock = self._get_async_lock()
            async with async_lock:
                if url in self._memory_cache:
                    del self._memory_cache[url]
        
        entry = self._load_from_disk(url)
        if self._is_valid_cache(entry):
            self.stats["disk_hits"] += 1
            return entry.content
        elif entry is not None:
            self._remove_from_disk(url)
        
        return await self._fetch_and_cache_async(url)

    def get_stats(self) -> dict:
        with self._cache_lock:
            memory_size = len(self._memory_cache)
        
        disk_size = 0
        if self.enable_disk_cache and self.cache_dir.exists():
            disk_size = len(list(self.cache_dir.glob("*.json")))
        
        return {
            **self.stats,
            "memory_cache_size": memory_size,
            "disk_cache_size": disk_size,
            "avg_fetch_time": (
                self.stats["total_fetch_time"] / self.stats["fetch_count"]
                if self.stats["fetch_count"] > 0 else 0.0
            ),
        }
    
    def reset_stats(self):
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "fetch_count": 0,
            "fetch_success": 0,
            "fetch_failed": 0,
            "total_fetch_time": 0.0,
        }
    
    def clear_memory_cache(self):
        with self._cache_lock:
            self._memory_cache.clear()
    
    def clear_disk_cache(self):
        if self.enable_disk_cache and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
    
    def close(self):
        self._jina_client.close()
    
    async def aclose(self):
        await self._jina_client.aclose()


def get_url_cache(
    cache_dir: str | Path | None = None,
    fetch_timeout: int = 10,
    max_cache_length: int = 30000,
    verbose: bool = False,
    jina_api_key: str | None = None,
) -> URLCacheManager:
    return URLCacheManager(
        cache_dir=cache_dir,
        fetch_timeout=fetch_timeout,
        max_cache_length=max_cache_length,
        verbose=verbose,
        jina_api_key=jina_api_key,
    )


def fetch_url(url: str) -> str:
    return get_url_cache().get(url)


def fetch_urls(urls: list[str]) -> dict[str, str]:
    return get_url_cache().get_batch(urls)


async def fetch_url_async(url: str) -> str:
    return await get_url_cache().get_async(url)


async def fetch_urls_async(urls: list[str]) -> dict[str, str]:
    return await get_url_cache().get_batch_async(urls)
