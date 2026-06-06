import asyncio
import paramiko
from langchain_openai import ChatOpenAI
from pyserxng import LocalSearXNGClient
from crawl4ai import *
from crawl4ai.extraction_strategy import NoExtractionStrategy
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from config_manager import server_config, tools_config
from logs.logging_setup import logger

class Searcher:
    """
    handles web searching and content extraction using searxng and crawl4ai.
    """
    def __init__(self):
        """
        initializes the searcher with configuration values.
        """
        self.host = server_config.fetch("search.host", "http://localhost:8888")
        self.url_ceil_surface = server_config.fetch("search.url_ceil_surface", 5)
        self.url_ceil_deep = server_config.fetch("search.url_ceil_deep", 9)
        self.bm25_threshold = server_config.fetch("search.bm25_threshold", 1.15)

        self.browser_cfg = BrowserConfig(
            headless = True,
            verbose = False,
            extra_args = server_config.fetch("browser.extra_args", []),
            headers = server_config.fetch("browser.headers", {}),
        )

        # initialize condenser logic within searcher
        m_config = server_config.fetch("settings", {})

        self.core_model = ChatOpenAI(
            model=m_config.get("model"),
            base_url=m_config.get("base_url"),
            temperature=m_config.get("temperature", 0.2),
            api_key=m_config.get("api_key")
        )

    async def ping(self) -> bool:
        """
        checks if the searxng host is reachable.

        returns:
            bool: true if reachable, false otherwise.
        """
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                # searxng usually has a /status or just checking the base url
                res = await client.get(self.host, timeout=2)
                return res.status_code == 200
        except Exception:
            return False

    async def condense(self, query: str, context: str, surface: bool = True) -> str:
        """
        condenses the given context based on the query and mode.

        args:
            query (str): the search query.
            context (str): raw web content to condense.
            surface (bool): true for basic summary, false for technical deep-dive.

        returns:
            str: condensed summary.
        """
        prompt_key = "condense.technical" if not surface else "condense.basic"
        prompt_template = tools_config.fetch(prompt_key)
        system_prompt = tools_config.fetch("condense.system")
        
        if not prompt_template:
            logger.error(f"prompt template {prompt_key} not found in tools_config.yaml")
            return context # fallback to raw content if prompt missing

        truncated_context = context[:15000]
        user_prompt = prompt_template.format(query=query, context=truncated_context)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            res = await self.core_model.ainvoke(messages)
            content = res.content
            if isinstance(content, list):
                # handle list of content blocks
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "".join(text_parts)
            return str(content) if content is not None else ""
        except Exception as e:
            logger.error(f"condensation error: {e}")
            return context # fallback to raw content on error

    
    def _search_sync(self, query: str):
        """
        performs a synchronous search using searxng.

        args:
            query (str): the search query.

        returns:
            list[str]: a list of urls found.
        """
        try:
            with LocalSearXNGClient(self.host) as client:
                return [str(r.url) for r in client.search(query).results]
        except Exception as e:
            logger.error(f"search error: {e}")
            return []
        

    async def _search(self, query: str):
        """
        asynchronously wraps the synchronous search.

        args:
            query (str): the search query.

        returns:
            list[str]: a list of urls found.
        """
        return await asyncio.to_thread(self._search_sync, query)
    

    async def get_links(self, query: str, surface: bool = False):
        """
        retrieves a list of links based on the query and depth.

        args:
            query (str): the search query.
            surface (bool): true for faster surface search, false for deep.

        returns:
            list[str]: a filtered list of urls.
        """
        # extract host domain to filter out internal results (like searxng about page)
        host_domain = self.host.split("//")[-1].split(":")[0]

        if surface:
            results = await self._search(query)
            return [url for url in results if host_domain not in url][:self.url_ceil_surface]

        # fetch query templates from config, fallback to single query if not found
        query_templates = server_config.fetch("search.queries", ["{query}"])
        queries = [template.format(query=query) for template in query_templates]

        results = await asyncio.gather(
            *(self._search(q) for q in queries)
        )

        final = []
        # first pass: take the top valid result from each unique query result set
        for batch in results:
            for url in batch:
                if host_domain not in url and url not in final:
                    final.append(url)
                    break # only take one from each batch in the first pass
        
        # second pass: fill up to ceil deep
        for batch in results:
            for url in batch:
                if len(final) >= self.url_ceil_deep:
                    break
                if host_domain not in url and url not in final:
                    final.append(url)
            if len(final) >= self.url_ceil_deep:
                break

        logger.info(f"gathered {len(final)} unique urls for crawling.")
        return final
    

    def _crawler_cfg(self, query: str, surface: bool):
        """
        generates the crawler configuration.

        args:
            query (str): the original search query for bm25 filtering.
            surface (bool): true for surface-level timeouts.

        returns:
            crawlerrunconfig: the configuration object for crawl4ai.
        """
        selector = None if surface else server_config.fetch("crawler.selectors.deep")
        timeout = server_config.fetch("crawler.page_timeout.surface", 5000) if surface else server_config.fetch("crawler.page_timeout.deep", 8000)

        return CrawlerRunConfig(
            cache_mode = CacheMode.BYPASS,
            excluded_tags = server_config.fetch("crawler.excluded_tags", []),
            css_selector = selector,
            remove_overlay_elements = True,
            extraction_strategy = NoExtractionStrategy(),
            page_timeout = timeout,
            markdown_generator = DefaultMarkdownGenerator(
                content_filter = BM25ContentFilter(
                    user_query = query,
                    bm25_threshold = self.bm25_threshold,
                ),
                options = server_config.fetch("crawler.markdown_options", {
                    "ignore_links": True,
                    "ignore_images": True,
                    "body_width": 0,
                }),
            ),
        )

    async def __call__(self, query: str, surface: bool = False) -> str:
        """
        executes the search and crawl pipeline.

        args:
            query (str): the search query.
            surface (bool): true for surface-level search, false for deep.

        returns:
            str: the concatenated markdown content extracted from results.
        """
        urls = await self.get_links(query, surface)
        if not urls:
            return "NO SOURCES FOUND."

        cfg = self._crawler_cfg(query, surface)

        # use memoryadaptivedispatcher with a ratelimiter for predictable performance
        dispatcher = MemoryAdaptiveDispatcher(
            max_session_permit=10,
            rate_limiter=RateLimiter(
                base_delay=(0.5, 1.0) # aim for ~1-2 requests per second
            )
        )

        async with AsyncWebCrawler(
            config = self.browser_cfg
        ) as crawler:

            results = await crawler.arun_many(
                urls = urls,
                config = cfg,
                dispatcher = dispatcher,
            )

        chunks = [
            r.markdown.fit_markdown.strip()
            for r in results
            if (
                r.success
                and r.markdown
                and len(r.markdown.fit_markdown.strip()) > 150
            )
        ]

        if not chunks:
            return "NO USEFUL CONTENT EXTRACTED."

        context = "\n\n---\n\n".join(chunks)
        
        # automatically condense the results
        result = await self.condense(query, context, surface)
        
        return result
         

class Recon:
    """
    provides tools for terminal execution and remote connectivity verification.
    """
    async def ping_ssh(self) -> dict:
        """
        verifies ssh connectivity and returns remote system specs.

        returns:
            dict: status and os info.
        """
        host = server_config.fetch("ssh.host")
        user = server_config.fetch("ssh.user")
        key_path = server_config.fetch("ssh.key_path")

        if not all([host, user, key_path]):
            return {"status": "failed", "os": None}

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            await asyncio.to_thread(
                ssh.connect,
                hostname = host,
                username = user,
                key_filename = key_path,
                timeout = 5
            )
            
            cmd = "grep 'PRETTY_NAME' /etc/os-release | cut -d'=' -f2 | tr -d '\"' && uname -m"
            stdin, stdout, stderr = await asyncio.to_thread(ssh.exec_command, cmd)
            os_info = (await asyncio.to_thread(stdout.read)).decode().strip().replace("\n", " ")
            ssh.close()
            
            if os_info:
                return {"status": "success", "os": os_info}
            else:
                return {"status": "failed", "os": None}
        except Exception as e:
            logger.error(f"ssh verification failed: {e}")
            return {"status": "failed", "os": None}

    async def terminal(self, command: str) -> str:
        """
        executes a command on the remote vm via ssh and returns output.

        args:
            command (str): the command to execute.

        returns:
            str: command output or error message.
        """
        host = server_config.fetch("ssh.host")
        user = server_config.fetch("ssh.user")
        key_path = server_config.fetch("ssh.key_path")
        prefix = server_config.fetch("terminal.prefix", "")

        if not all([host, user, key_path]):
            return "error: ssh configuration incomplete."

        full_command = f"{prefix} {command}".strip()
        logger.info(f"executing remote: {full_command}")

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            await asyncio.to_thread(
                ssh.connect,
                hostname = host,
                username = user,
                key_filename = key_path,
                timeout = 10
            )

            # use standard exec_command without pty for cleaner raw output
            stdin, stdout, stderr = await asyncio.to_thread(ssh.exec_command, full_command)
            
            # read stdout and stderr fully
            out_data = (await asyncio.to_thread(stdout.read)).decode('utf-8', 'replace')
            err_data = (await asyncio.to_thread(stderr.read)).decode('utf-8', 'replace')

            # capture the exit status to classify the error stream
            exit_status = await asyncio.to_thread(stdout.channel.recv_exit_status)
            ssh.close()
            
            res = out_data
            if err_data.strip():
                # classify stderr based on official exit code:
                # 0 = system notices / progress / warnings
                # != 0 = actual command failure
                label = "[ERROR]" if exit_status != 0 else "[SYSTEM_INFO]"
                res += f"\n{label}\n{err_data}"
            
            return res

        except Exception as e:
            msg = f"error: ssh execution failed: {str(e)}"
            logger.error(msg)
            return msg
