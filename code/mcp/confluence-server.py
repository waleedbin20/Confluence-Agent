import logging
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Any, Union
import time
import aiohttp
from atlassian import Confluence
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")

mcp = FastMCP("Demo")

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(asctime)s %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Rate limiting configuration
CALLS_PER_MINUTE = 30
RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 10

# Input validation models
class ConfluenceBase(BaseModel):
    confluence_url: str = Field(..., description="The Confluence instance URL")
    access_token: str = Field(..., description="The API access token")

class SearchInput(ConfluenceBase):
    query: str = Field(..., description="The search query string")

class SpaceInput(ConfluenceBase):
    pass

class PageInput(ConfluenceBase):
    space_key: str = Field(..., description="The space key where the page exists")
    page_id: str = Field(..., description="The ID of the page")

class CreatePageInput(ConfluenceBase):
    space_key: str = Field(..., description="The space key where to create the page")
    title: str = Field(..., description="The title of the new page")
    content: str = Field(..., description="The HTML content of the page")

class UpdatePageInput(PageInput):
    content: str = Field(..., description="The new HTML content")

# Initialize MCP server
mcp = FastMCP("Confluence MCP")

# Decorator for rate limiting and retries
def rate_limited_retry(func):
    @sleep_and_retry
    @limits(calls=CALLS_PER_MINUTE, period=60)
    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT)
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed", extra={
                "function": func.__name__,
                "duration": duration,
                "status": "success"
            })
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed: {str(e)}", extra={
                "function": func.__name__,
                "duration": duration,
                "error": str(e),
                "status": "error"
            })
            raise
    return wrapper

def create_confluence_client(url: str, token: str) -> Confluence:
    """Create and configure Confluence client with timeout and retry settings."""
    return Confluence(
        url=url,
        username="pawan.kumar@arelis.digital",
        password=token,
        cloud=True,
        timeout=30
    )

@mcp.tool()
async def list_tools() -> Dict[str, Any]:
    """List all available MCP tools.
    
    This tool provides information about all available Confluence operations,
    including their parameters and expected return values.
    
    Returns:
        Dict containing list of available tools with their schemas
    
    Example:
        ```python
        tools = await list_tools()
        for tool in tools['tools']:
            print(f"Tool: {tool['name']}")
            print(f"Description: {tool['description']}")
        ```
    """
    return {
        "tools": [
            {
                "name": "confluence_search",
                "description": "Search for content in Confluence using CQL",
                "inputSchema": SearchInput.model_json_schema(),
                "examples": [
                    {
                        "description": "Search for pages containing 'architecture'",
                        "code": 'await confluence_search(confluence_url="https://your-domain.atlassian.net/wiki", access_token="your-token", query="architecture")'
                    }
                ]
            },
            {
                "name": "confluence_get_spaces",
                "description": "Get list of available Confluence spaces",
                "inputSchema": SpaceInput.model_json_schema(),
                "examples": [
                    {
                        "description": "List all available spaces",
                        "code": 'await confluence_get_spaces(confluence_url="https://your-domain.atlassian.net/wiki", access_token="your-token")'
                    }
                ]
            },
            {
                "name": "confluence_get_page",
                "description": "Get detailed content of a Confluence page",
                "inputSchema": PageInput.model_json_schema(),
                "examples": [
                    {
                        "description": "Get page content by ID",
                        "code": 'await confluence_get_page(confluence_url="https://your-domain.atlassian.net/wiki", access_token="your-token", space_key="SPACE", page_id="123456")'
                    }
                ]
            },
            {
                "name": "confluence_create_page",
                "description": "Create a new page in Confluence",
                "inputSchema": CreatePageInput.model_json_schema(),
                "examples": [
                    {
                        "description": "Create a new page",
                        "code": 'await confluence_create_page(confluence_url="https://your-domain.atlassian.net/wiki", access_token="your-token", space_key="SPACE", title="New Page", content="<p>Hello World</p>")'
                    }
                ]
            },
            {
                "name": "confluence_update_page",
                "description": "Update content of an existing Confluence page",
                "inputSchema": UpdatePageInput.model_json_schema(),
                "examples": [
                    {
                        "description": "Update page content",
                        "code": 'await confluence_update_page(confluence_url="https://your-domain.atlassian.net/wiki", access_token="your-token", space_key="SPACE", page_id="123456", content="<p>Updated content</p>")'
                    }
                ]
            },
            {
                "name": "confluence_delete_page",
                "description": "Delete a Confluence page",
                "inputSchema": PageInput.model_json_schema(),
                "examples": [
                    {
                        "description": "Delete a page",
                        "code": 'await confluence_delete_page(confluence_url="https://your-domain.atlassian.net/wiki", access_token="your-token", space_key="SPACE", page_id="123456")'
                    }
                ]
            }
        ]
    }

@mcp.tool()
@rate_limited_retry
async def confluence_search(confluence_url: str, access_token: str, query: str) -> Dict[str, Any]:
    """Search for content in Confluence using CQL.
    
    This tool searches Confluence content using the provided query string.
    The search uses Confluence Query Language (CQL) for advanced searching capabilities.
    
    Args:
        confluence_url: The Confluence instance URL
        access_token: The API access token
        query: The search query string
    
    Returns:
        Dict containing search results with titles, URLs and metadata
    
    Example:
        ```python
        results = await confluence_search(
            confluence_url="https://your-domain.atlassian.net/wiki",
            access_token="your-token",
            query="project in (SD)"
        )
        ```
    """
    try:
        # Validate input
        input_data = SearchInput(
            confluence_url=confluence_url,
            access_token=access_token,
            query=query
        )
        
        # Create client
        confluence = create_confluence_client(input_data.confluence_url, input_data.access_token)
        
        # Execute search with progress logging
        logger.info(f"Executing search query: {input_data.query}")
        results = confluence.cql(f'text ~ "{input_data.query}"')
        
        formatted_results = []
        for result in results.get('results', []):
            space_key = result.get('space', {}).get('key', '')
            content_id = result.get('content', {}).get('id', '')
            title = result.get('content', {}).get('title', '')
            
            formatted_results.append({
                "id": f"confluence://{space_key}/{content_id}",
                "title": title,
                "space_key": space_key,
                "content_id": content_id,
                "type": result.get('content', {}).get('type', ''),
                "url": result.get('_links', {}).get('webui', '')
            })
        
        logger.info(f"Search completed, found {len(formatted_results)} results")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Found {len(formatted_results)} results"
                },
                {
                    "type": "json",
                    "data": {"results": formatted_results}
                }
            ]
        }
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }
            ]
        }

@mcp.tool()
@rate_limited_retry
async def confluence_get_spaces(confluence_url: str, access_token: str) -> Dict[str, Any]:
    """Get list of available Confluence spaces.
    
    This tool retrieves all accessible Confluence spaces for the authenticated user.
    
    Args:
        confluence_url: The Confluence instance URL
        access_token: The API access token
    
    Returns:
        Dict containing list of spaces with their details
    
    Example:
        ```python
        spaces = await confluence_get_spaces(
            confluence_url="https://your-domain.atlassian.net/wiki",
            access_token="your-token"
        )
        ```
    """
    try:
        # Validate input
        input_data = SpaceInput(
            confluence_url=confluence_url,
            access_token=access_token
        )
        
        # Create client
        confluence = create_confluence_client(input_data.confluence_url, input_data.access_token)
        
        # Get spaces with progress logging
        logger.info("Retrieving Confluence spaces")
        spaces_response = confluence.get_all_spaces()
        spaces = spaces_response.get('results', [])
        
        formatted_spaces = []
        for space in spaces:
            space_key = space.get('key', '')
            formatted_spaces.append({
                "id": f"confluence://{space_key}",
                "key": space_key,
                "name": space.get('name', ''),
                "type": space.get('type', ''),
                "url": space.get('_links', {}).get('webui', ''),
                "description": space.get('description', {}).get('plain', {}).get('value', '')
            })
        
        logger.info(f"Retrieved {len(formatted_spaces)} spaces")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Found {len(formatted_spaces)} spaces"
                },
                {
                    "type": "json",
                    "data": {"spaces": formatted_spaces}
                }
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get spaces: {str(e)}")
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }
            ]
        }

@mcp.tool()
@rate_limited_retry
async def confluence_get_page(confluence_url: str, access_token: str, space_key: str, page_id: str) -> Dict[str, Any]:
    """Get detailed content of a Confluence page.
    
    This tool retrieves the complete content and metadata of a specific Confluence page.
    
    Args:
        confluence_url: The Confluence instance URL
        access_token: The API access token
        space_key: The space key where the page exists
        page_id: The ID of the page to retrieve
    
    Returns:
        Dict containing page content and metadata
    
    Example:
        ```python
        page = await confluence_get_page(
            confluence_url="https://your-domain.atlassian.net/wiki",
            access_token="your-token",
            space_key="SPACE",
            page_id="123456"
        )
        ```
    """
    try:
        # Validate input
        input_data = PageInput(
            confluence_url=confluence_url,
            access_token=access_token,
            space_key=space_key,
            page_id=page_id
        )
        
        # Create client
        confluence = create_confluence_client(input_data.confluence_url, input_data.access_token)
        
        # Get page with progress logging
        logger.info(f"Retrieving page {input_data.page_id} from space {input_data.space_key}")
        content = confluence.get_page_by_id(
            input_data.page_id,
            expand='body.storage,version,space'
        )
        
        page_data = {
            "title": content.get('title', ''),
            "content": content.get('body', {}).get('storage', {}).get('value', ''),
            "version": content.get('version', {}).get('number', 1),
            "space_key": content.get('space', {}).get('key', ''),
            "last_modified": content.get('version', {}).get('when', ''),
            "author": content.get('version', {}).get('by', {}).get('displayName', '')
        }
        
        logger.info(f"Retrieved page: {page_data['title']}")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Retrieved page: {page_data['title']}"
                },
                {
                    "type": "json",
                    "data": page_data
                }
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get page: {str(e)}")
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }
            ]
        }

@mcp.tool()
@rate_limited_retry
async def confluence_create_page(confluence_url: str, access_token: str, space_key: str, title: str, content: str) -> Dict[str, Any]:
    """Create a new page in Confluence.
    
    This tool creates a new page in the specified Confluence space.
    
    Args:
        confluence_url: The Confluence instance URL
        access_token: The API access token
        space_key: The space key where to create the page
        title: The title of the new page
        content: The HTML content of the page
    
    Returns:
        Dict containing the created page details
    
    Example:
        ```python
        new_page = await confluence_create_page(
            confluence_url="https://your-domain.atlassian.net/wiki",
            access_token="your-token",
            space_key="SPACE",
            title="New Page",
            content="<p>Hello World</p>"
        )
        ```
    """
    try:
        # Validate input
        input_data = CreatePageInput(
            confluence_url=confluence_url,
            access_token=access_token,
            space_key=space_key,
            title=title,
            content=content
        )
        
        # Create client
        confluence = create_confluence_client(input_data.confluence_url, input_data.access_token)
        
        # Create page with progress logging
        logger.info(f"Creating page '{input_data.title}' in space {input_data.space_key}")
        page = confluence.create_page(
            space=input_data.space_key,
            title=input_data.title,
            body=input_data.content
        )
        
        page_data = {
            "id": f"confluence://{input_data.space_key}/{page['id']}",
            "title": page['title'],
            "space_key": input_data.space_key,
            "content_id": page['id'],
            "url": page.get('_links', {}).get('webui', '')
        }
        
        logger.info(f"Created page: {page_data['title']}")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Created page: {page_data['title']}"
                },
                {
                    "type": "json",
                    "data": page_data
                }
            ]
        }
    except Exception as e:
        logger.error(f"Failed to create page: {str(e)}")
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }
            ]
        }

@mcp.tool()
@rate_limited_retry
async def confluence_update_page(confluence_url: str, access_token: str, space_key: str, page_id: str, content: str) -> Dict[str, Any]:
    """Update content of an existing Confluence page.
    
    This tool updates the content of an existing page while preserving its metadata.
    
    Args:
        confluence_url: The Confluence instance URL
        access_token: The API access token
        space_key: The space key where the page exists
        page_id: The ID of the page to update
        content: The new HTML content
    
    Returns:
        Dict indicating success or failure
    
    Example:
        ```python
        result = await confluence_update_page(
            confluence_url="https://your-domain.atlassian.net/wiki",
            access_token="your-token",
            space_key="SPACE",
            page_id="123456",
            content="<p>Updated content</p>"
        )
        ```
    """
    try:
        # Validate input
        input_data = UpdatePageInput(
            confluence_url=confluence_url,
            access_token=access_token,
            space_key=space_key,
            page_id=page_id,
            content=content
        )
        
        # Create client
        confluence = create_confluence_client(input_data.confluence_url, input_data.access_token)
        
        # Update page with progress logging
        logger.info(f"Updating page {input_data.page_id} in space {input_data.space_key}")
        page = confluence.get_page_by_id(input_data.page_id)
        
        confluence.update_page(
            page_id=input_data.page_id,
            title=page['title'],
            body=input_data.content,
            minor_edit=True
        )
        
        logger.info(f"Updated page: {page['title']}")
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Page updated successfully"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Failed to update page: {str(e)}")
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }
            ]
        }

@mcp.tool()
@rate_limited_retry
async def confluence_delete_page(confluence_url: str, access_token: str, space_key: str, page_id: str) -> Dict[str, Any]:
    """Delete a Confluence page.
    
    This tool permanently deletes a page from Confluence.
    
    Args:
        confluence_url: The Confluence instance URL
        access_token: The API access token
        space_key: The space key where the page exists
        page_id: The ID of the page to delete
    
    Returns:
        Dict indicating success or failure
    
    Example:
        ```python
        result = await confluence_delete_page(
            confluence_url="https://your-domain.atlassian.net/wiki",
            access_token="your-token",
            space_key="SPACE",
            page_id="123456"
        )
        ```
    """
    try:
        # Validate input
        input_data = PageInput(
            confluence_url=confluence_url,
            access_token=access_token,
            space_key=space_key,
            page_id=page_id
        )
        
        # Create client
        confluence = create_confluence_client(input_data.confluence_url, input_data.access_token)
        
        # Delete page with progress logging
        logger.info(f"Deleting page {input_data.page_id} from space {input_data.space_key}")
        confluence.remove_page(input_data.page_id)
        
        logger.info("Page deleted successfully")
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Page deleted successfully"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Failed to delete page: {str(e)}")
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }
            ]
        }

if __name__ == "__main__":
    mcp.run()