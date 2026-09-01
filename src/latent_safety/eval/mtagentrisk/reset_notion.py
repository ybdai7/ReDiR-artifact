import sys
import os
import logging
import re
import time
from pathlib import Path
from typing import Tuple

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = Path(current_dir).resolve().parents[3]
workspace_root = repo_root.parent
project_candidates = [
    repo_root / "third_party" / "mcpmark",
    Path(current_dir).resolve().parent / "mcpmark-main",
    workspace_root / "ToolShield" / "mcpmark-main",
    workspace_root / "mcpmark-main",
]
project_root = next(
    (str(path) for path in project_candidates if (path / "src").exists()),
    os.path.abspath(os.path.join(current_dir, "..")),
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORTS ---
try:
    from src.mcp_services.notion.notion_state_manager import NotionStateManager
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import Notion modules: {e}")
    sys.exit(1)

from playwright.sync_api import Browser, BrowserContext, sync_playwright

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("NotionReset")

ORPHAN_SOURCE_TITLE_RE = re.compile(r".+\s+\(\d+\)$")


class SplitNetworkNotionStateManager(NotionStateManager):
    """Keep Notion API traffic direct while optionally proxying browser UI traffic.

    Notion's API and its browser application have shown different reliability on this
    host.  The reset launcher removes process-wide proxy variables so the API clients
    use the stable direct route.  Playwright receives an explicit browser-only proxy,
    avoiding both API SSL EOF failures and direct browser navigation timeouts.
    """

    def _ensure_browser(self) -> Tuple[Browser, BrowserContext]:
        if self._playwright is None:
            self._playwright = sync_playwright().start()

        if self._browser is None:
            browser_type = getattr(self._playwright, self.browser_name)
            launch_options = {"headless": self.headless}
            browser_proxy = os.environ.get(
                "MTAGENTRISK_NOTION_BROWSER_PROXY", ""
            ).strip()
            if browser_proxy:
                launch_options["proxy"] = {"server": browser_proxy}
                logger.info("Notion Playwright browser proxy: %s", browser_proxy)
            else:
                logger.info("Notion Playwright browser proxy: disabled")
            self._browser = browser_type.launch(**launch_options)

        if self._context is None:
            self._context = self._browser.new_context(
                storage_state=str(self.state_file),
                locale="en-US",
            )

        return self._browser, self._context

    def _duplicate_current_initial_state_to_eval_via_runtime_queue(
        self,
        page,
        *,
        original_initial_state_id: str,
        original_initial_state_title: str,
        new_title,
    ) -> str:
        """Skip the stale Notion frontend-module duplication path.

        MCPMark's runtime queue currently depends on webpack module IDs from an
        older Notion build.  The current application still supports the visible
        Duplicate action, while waiting for those stale IDs costs 90 seconds per
        page before the working UI fallback is attempted.  Raising immediately
        preserves the upstream fallback and avoids treating expected frontend
        drift as a network hang.
        """
        raise RuntimeError(
            "Notion runtime queue disabled: current frontend module IDs are unstable"
        )

    def _duplicate_current_initial_state_via_runtime(
        self,
        page,
        *,
        original_initial_state_id: str,
        original_initial_state_title: str,
        wait_timeout: int,
    ) -> str:
        """Fail fast if the visible Duplicate action was not available.

        This second runtime fallback uses the same stale module loader.  Let the
        manager reopen the source page for its normal retry instead of spending
        another 90 seconds on a path that cannot succeed with the current build.
        """
        raise RuntimeError(
            "Notion legacy runtime duplicate disabled: retry the browser UI path"
        )

    def _get_current_runtime_page_id(self, page):
        """Read the current page ID from its canonical URL without stale modules.

        The upstream UI-result polling calls this method before checking the
        public API for the newly created page.  Its runtime implementation waits
        30 seconds on every poll with the current Notion bundle, so URL parsing is
        both faster and sufficient for detecting browser navigation.
        """
        try:
            return self._extract_initial_state_id_from_url(page.url)
        except (AttributeError, TypeError, ValueError):
            return None

    def _move_current_page_to_env_via_runtime(
        self,
        page,
        *,
        duplicated_initial_state_id: str,
        wait_timeout: int,
    ) -> None:
        """Do not enter the stale runtime move after a real UI move failure."""
        raise RuntimeError(
            "Notion runtime move disabled: current frontend module IDs are unstable"
        )

    def _rename_initial_state_via_api(
        self, initial_state_id: str, new_title: str
    ) -> None:
        """Wait for the moved page to become visible to the eval integration.

        Notion's UI commits the move before the API permission change is visible.
        The upstream implementation performs one immediate PATCH, so a transient
        404 makes the caller duplicate the whole page again.  Polling here keeps
        the existing UI move and turns permission propagation into a bounded wait.
        """
        deadline = time.monotonic() + 120
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.eval_notion_client.pages.update(
                    page_id=initial_state_id,
                    properties={
                        "title": {
                            "title": [{"text": {"content": new_title}}]
                        }
                    },
                )
                logger.info(
                    "| ✓ Eval integration can access moved page: %s",
                    initial_state_id,
                )
                return
            except Exception as exc:
                last_error = exc
                time.sleep(5)
        raise RuntimeError(
            f"Moved page did not become visible to eval integration: {last_error}"
        )

    def _duplicate_initial_state_for_task(
        self,
        initial_state_url: str,
        category: str,
        task_name: str,
        *,
        max_retries: int = 2,
        initial_wait_ms: int = 180_000,
    ) -> Tuple[str, str]:
        """Use the upstream retry loop without sleeping after its final failure."""
        if not self.state_file.exists():
            raise FileNotFoundError(
                "Authentication state 'notion_state.json' not found. "
                "Run the Notion login helper first."
            )

        last_exc = None
        for attempt in range(max_retries + 1):
            wait_timeout = initial_wait_ms * (attempt + 1)
            page = None
            try:
                _, context = self._ensure_browser()
                page = context.new_page()
                logger.info("| ○ Navigating to initial state for %s...", category)
                start_time = time.time()
                initial_state_id = self._extract_initial_state_id_from_url(
                    initial_state_url
                )
                canonical_url = self._resolve_page_url(
                    self.source_notion_client, initial_state_id
                )
                if canonical_url:
                    initial_state_url = canonical_url
                page.goto(
                    initial_state_url,
                    wait_until="domcontentloaded",
                    timeout=120_000,
                )
                self._wait_for_notion_page_ready(
                    page,
                    expected_title=self._category_to_initial_state_title(category),
                    timeout=60_000,
                )
                context.storage_state(path=str(self.state_file))
                initial_state_title = self._category_to_initial_state_title(category)
                duplicated_id = self._duplicate_current_initial_state(
                    page,
                    new_title=initial_state_title,
                    original_initial_state_id=initial_state_id,
                    original_initial_state_title=initial_state_title,
                    wait_timeout=wait_timeout,
                )
                context.storage_state(path=str(self.state_file))
                logger.info(
                    "| ✓ Initial state duplicated successfully in %.2f seconds (task: %s).",
                    time.time() - start_time,
                    task_name,
                )
                return page.url, duplicated_id
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "| ✗ Duplication attempt %d failed: %s. Retrying...",
                        attempt + 1,
                        exc,
                    )
                    time.sleep(120 * (attempt + 1))
            finally:
                if page:
                    try:
                        page.close()
                    except Exception:
                        pass

        raise RuntimeError(
            f"Initial state duplication failed for task '{task_name}' "
            f"after {max_retries + 1} attempts: {last_exc}"
        )

def env_bool(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def api_call_with_retry(label, function, *args, attempts=5, **kwargs):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(5 * attempt, 20)
            logger.warning(
                "Notion API call failed (%s), retry %d/%d in %ds: %s",
                label,
                attempt,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    raise RuntimeError(f"Notion API call failed after {attempts} attempts ({label}): {last_error}")

def find_ghost_page(client, parent_id, expected_title):
    """
    Manually searches for a page that might have been created but missed by the verifier.
    """
    try:
        # List children of the Source Hub
        response = client.blocks.children.list(block_id=parent_id)
        results = response.get("results", [])
        
        for block in results:
            if block["type"] == "child_page":
                title = block["child_page"]["title"]
                # Check if it matches "Title (1)" or just "Title" (sometimes Notion duplicates oddly)
                if title == expected_title:
                    return block["id"]
    except Exception as e:
        logger.error(f"     ⚠️ Ghost search failed: {e}")
    return None

def full_reset():
    s_key = os.environ.get("SOURCE_NOTION_KEY")
    e_key = os.environ.get("NOTION_TOKEN")
    browser = os.environ.get("PLAYWRIGHT_BROWSER", "firefox")
    headless = env_bool("PLAYWRIGHT_HEADLESS", True)
    source_parent_page_title = os.environ.get("SOURCE_PARENT_PAGE_TITLE", "MCPMark Source Hub")
    eval_parent_page_title = os.environ.get("EVAL_PARENT_PAGE_TITLE", "MCPMark Eval Hub")
    resume = env_bool("MTAGENTRISK_NOTION_RESET_RESUME", False)
    
    if not s_key or not e_key:
        logger.error("❌ Error: Missing API Keys.")
        sys.exit(1)

    try:
        manager = SplitNetworkNotionStateManager(
            source_notion_key=s_key,
            eval_notion_key=e_key,
            headless=headless,
            browser=browser,
            source_parent_page_title=source_parent_page_title,
            eval_parent_page_title=eval_parent_page_title,
        )
        source_client = manager.source_notion_client
        eval_client = manager.eval_notion_client # Use this for moving pages
        
        source_hub_id = manager._ensure_source_hub_page_id()
        eval_hub_id = manager._ensure_eval_parent_page_id()

        if not source_hub_id or not eval_hub_id:
            logger.error("❌ Critical Error: Could not find Source or Eval Hub pages.")
            return False

        logger.info("🔄 STARTING FULL RESET")

        # 1. WIPE EVAL HUB, or preserve verified pages after a transient crash.
        existing_titles = set()
        children = api_call_with_retry(
            "list Eval Hub children",
            eval_client.blocks.children.list,
            block_id=eval_hub_id,
        ).get("results", [])
        if resume:
            existing_titles = {
                block.get("child_page", {}).get("title", "").strip()
                for block in children
                if block.get("type") == "child_page"
            }
            existing_titles.discard("")
            logger.info(
                "♻️  Resuming reset with %d existing Eval Hub page(s): %s",
                len(existing_titles),
                ", ".join(sorted(existing_titles)),
            )
        else:
            logger.info("🗑️  Wiping Eval Hub...")
            for block in children:
                block_id = block["id"]
                block_type = block.get("type")

                try:
                    if block_type == "child_page":
                        api_call_with_retry(
                            f"archive Eval page {block_id}",
                            eval_client.pages.update,
                            page_id=block_id,
                            archived=True,
                        )
                    else:
                        api_call_with_retry(
                            f"archive Eval block {block_id}",
                            eval_client.blocks.update,
                            block_id=block_id,
                            archived=True,
                        )
                    print(f"   - Deleted: {block_id} ({block_type})")
                except Exception as e:
                    logger.warning(f"   - Failed to delete {block_id}: {e}")

        # 2. CLONE SOURCE HUB
        logger.info("📋 Cloning Source Hub content...")
        try:
            cleaned = manager._cleanup_source_hub_orphans()
            if cleaned:
                logger.info("🧹 Cleaned %d source-hub orphan page(s) before cloning.", cleaned)
        except Exception as cleanup_err:
            logger.warning("⚠️ Source-hub orphan cleanup failed; will skip orphan-like titles: %s", cleanup_err)

        source_children = api_call_with_retry(
            "list Source Hub children",
            source_client.blocks.children.list,
            block_id=source_hub_id,
        ).get("results", [])
        cloned_count = 0
        failed_titles = []

        for child in source_children:
            if child["type"] == "child_page":
                title = child["child_page"]["title"]
                if ORPHAN_SOURCE_TITLE_RE.match(title.strip()):
                    logger.warning("   - Skipping source-hub orphan page '%s'", title)
                    continue
                if title.strip() in existing_titles:
                    logger.info("   - Reusing verified Eval Hub page '%s'", title)
                    cloned_count += 1
                    continue
                page_obj = api_call_with_retry(
                    f"retrieve source page {title}",
                    source_client.pages.retrieve,
                    page_id=child["id"],
                )
                source_url = page_obj.get("url")
                if not source_url:
                    logger.error(f"     ❌ Could not resolve URL for source page '{title}'")
                    continue
                
                logger.info(f"   - Cloning '{title}'...")
                
                # We need to capture the ID of the new page
                new_page_id = None
                
                try:
                    # Attempt standard duplication
                    # We pass 'title' as category to try and match casing, but the Manager often lowercases it anyway.
                    dup_url, dup_id = manager._duplicate_initial_state_for_task(
                        source_url, title, title
                    )
                    new_page_id = dup_id

                except Exception:
                    logger.warning("     ⚠️ Standard verification failed. Checking for 'Ghost Page'...")
                    
                    # RECOVERY: Search for the page manually with EXACT casing
                    expected_ghost_title = f"{title} (1)" 
                    new_page_id = find_ghost_page(source_client, source_hub_id, expected_ghost_title)
                    
                    if new_page_id:
                        logger.info(f"     👻 FOUND GHOST PAGE: {new_page_id}")
                    else:
                        logger.error(f"     ❌ Failed to find duplicated page '{expected_ghost_title}'")
                        failed_titles.append(title)

                # 3. POST-PROCESS: Move to Eval Hub and Rename
                if new_page_id:
                    try:
                        page_obj = api_call_with_retry(
                            f"verify Eval page {title}",
                            eval_client.pages.retrieve,
                            page_id=new_page_id,
                        )
                        parent = page_obj.get("parent", {}) if isinstance(page_obj, dict) else {}
                        if parent.get("page_id") != eval_hub_id:
                            raise RuntimeError(
                                f"Duplicated page is not under Eval Hub: parent={parent}"
                            )
                        print("     📦 Verified in Eval Hub")
                        print("     ✅ Success!")
                        cloned_count += 1
                        
                    except Exception as move_err:
                        logger.error(f"     ⚠️ Created but failed to move/rename: {move_err}")
                        failed_titles.append(title)

        logger.info(f"✅ Full Reset Complete. Cloned {cloned_count} pages.")
        if failed_titles:
            logger.error(
                "❌ Full Reset failed for %d page(s): %s",
                len(failed_titles),
                ", ".join(failed_titles),
            )
            return False
        eval_hub_url = manager._resolve_page_url(eval_client, eval_hub_id)
        print(eval_hub_url or eval_hub_id)
        return True

    except Exception as e:
        logger.error(f"❌ Script Crash: {e}")
        sys.exit(1)
    finally:
        try:
            manager.close()
        except Exception:
            pass

if __name__ == "__main__":
    if not full_reset():
        sys.exit(1)
