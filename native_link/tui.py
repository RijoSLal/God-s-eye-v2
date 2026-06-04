import sys
import os
from pathlib import Path

# signal that we are in TUI mode to avoid logging to console
os.environ["GODS_EYE_TUI"] = "true"

# add root directory to sys.path to enable imports
root_path = Path(__file__).parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from textual.app import App, ComposeResult
from textual.widgets import Input, Static, TextArea, Label
from textual.containers import Horizontal, ScrollableContainer, Vertical, Container
from textual.reactive import reactive
from textual import events
from textual.binding import Binding
from textual.widget import Widget
from rich.text import Text
from rich.markup import escape
from datetime import datetime
import asyncio
import random
import orjson
from telemetry import Monitor

from supervisor import Reporter

class EditorTextArea(TextArea):
    """
    json and markdown editor text area
    
    overrides key events to catch save and escape before default handling
    """
    async def _on_key(self, event: events.Key) -> None:
        """
        handles key events for the editor.

        args:
            event (events.Key): the key event.

        returns:
            none
        """
        if event.key in ("ctrl+s", "escape"):
            event.prevent_default()
            event.stop()
            for ancestor in self.ancestors:
                if hasattr(ancestor, "action_save") and hasattr(ancestor, "action_exit_editor"):
                    if event.key == "ctrl+s":
                        ancestor.action_save()
                    else:
                        ancestor.action_exit_editor()
                    return
        else:
            await super()._on_key(event)


class HistoryInput(TextArea):
    """
    text area with history support for chat input.
    """
    BINDINGS = [
        Binding("ctrl+v", "paste", "Paste", priority=True),
        Binding("up", "history_prev", "Prev", priority=True),
        Binding("down", "history_next", "Next", priority=True),
    ]

    def __init__(self, *args, **kwargs):
        """
        initializes the history input widget.

        args:
            *args: positional arguments for textarea.
            **kwargs: keyword arguments for textarea.

        returns:
            none
        """
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._hist_idx: int = -1
        self._draft: str = ""
        self.show_line_numbers = False

    def push_history(self, text: str) -> None:
        """
        adds a new entry to the command history.

        args:
            text (str): the text to add.

        returns:
            none
        """
        if text.strip() and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._hist_idx = -1

    def action_history_prev(self) -> None:
        """
        navigates to the previous item in history.

        returns:
            none
        """
        if self.cursor_location[0] == 0:
            if not self._history: return
            if self._hist_idx == -1:
                self._draft = self.text
                self._hist_idx = len(self._history) - 1
            elif self._hist_idx > 0:
                self._hist_idx -= 1
            self.text = self._history[self._hist_idx]
            lines = self.document.lines
            self.move_cursor((len(lines) - 1, len(lines[-1])))

    def action_history_next(self) -> None:
        """
        navigates to the next item in history.

        returns:
            none
        """
        lines = self.document.lines
        if self.cursor_location[0] == len(lines) - 1:
            if self._hist_idx == -1: return
            if self._hist_idx < len(self._history) - 1:
                self._hist_idx += 1
                self.text = self._history[self._hist_idx]
            else:
                self._hist_idx = -1
                self.text = self._draft
            lines = self.document.lines
            self.move_cursor((len(lines) - 1, len(lines[-1])))

    async def on_key(self, event: events.Key) -> None:
        """
        handles key events specifically for the enter key.

        args:
            event (events.Key): the key event.

        returns:
            none
        """
        if event.key == "enter":
            event.prevent_default()
            self.post_message(Input.Submitted(Input(), self.text))


monitor = Monitor()

WELCOME = r"""
  ██████╗  ██████╗ ██████╗ ██╗ ███████╗  ███████╗██╗   ██╗███████╗
 ██╔════╝ ██╔═══██╗██╔══██╗    ██╔════╝  ██╔════╝╚██╗ ██╔╝██╔════╝
 ██║  ███╗██║   ██║██║  ██║    ███████╗  █████╗   ╚████╔╝ █████╗
 ██║   ██║██║   ██║██║  ██║    ╚════██║  ██╔══╝    ╚██╔╝  ██╔══╝
 ╚██████╔╝╚██████╔╝██████╔╝    ███████║  ███████╗   ██║   ███████╗
  ╚═════╝  ╚═════╝ ╚═════╝     ╚══════╝   ══════╝   ╚═╝   ╚══════╝

         ◈  All-seeing · All-exploiting · Always watching  ◈
                           v2.0.0
"""


# ------------------------------------------------------------------
#  LEFT PANEL
# ------------------------------------------------------------------

class SystemStats(Widget):
    """
    widget to display system resource statistics.
    """
    stats_string = reactive("")

    DEFAULT_CSS = """
    SystemStats {
        height: 3;
        background: #0a0a0a;
        color: #c8c8c8;
        content-align: center middle;
    }
    """

    def on_mount(self) -> None:
        """
        initializes the periodic stats update.

        returns:
            none
        """
        self.set_interval(5.0, self.update_stats)

    def update_stats(self) -> None:
        """
        fetches and updates system stats.

        returns:
            none
        """
        try:
            stats = monitor.system_stats()
            if stats:
                cpu, ram, gpu, net, disk = stats
                self.stats_string = f"{cpu} | {gpu} | {ram} | {net} | {disk}"
            else:
                self.stats_string = ""
        except Exception:
            self.stats_string = ""

    def render(self) -> str:
        """
        renders the stats string.

        returns:
            str: the formatted stats.
        """
        return self.stats_string


class LeftPanel(Widget):
    """
    the left panel containing the remote terminal log and system stats.
    """
    DEFAULT_CSS = """
    LeftPanel {
        width: 1fr;
        height: 1fr;
        background: #060606;
        border-right: solid #1a1a1a;
    }
    #terminal-scroll {
        height: 1fr;
        background: #060606;
        scrollbar-size: 0 0;
        padding: 0 1;
    }
    #terminal-log {
        height: auto;
        width: 100%;
        layout: vertical;
    }
    .terminal-entry {
        width: 100%;
        height: auto;
        color: #888888;
        margin-bottom: 0;
        padding: 0;
    }
    """

    def _get_prompt(self) -> str:
        """
        generates the remote terminal prompt string.

        returns:
            str: the prompt with rich markup.
        """
        return "\n[bold #aaaaaa]remote[/]:[#666666]~[/][bold #aaaaaa]$ [/]"

    def compose(self) -> ComposeResult:
        """
        composes the left panel layout.

        returns:
            ComposeResult: the composed widgets.
        """
        with Vertical():
            with ScrollableContainer(id="terminal-scroll"):
                yield Vertical(id="terminal-log")
            yield SystemStats()

    def on_mount(self) -> None:
        """
        initializes the terminal log and polling.

        returns:
            none
        """
        self._prompt_text = self._get_prompt()
        self._active_prompt_widget = None
        self._show_prompt()
        self.set_interval(0.1, self.poll_terminal)

    def _show_prompt(self) -> None:
        """
        mounts a new prompt widget and keeps track of it.

        returns:
            none
        """
        try:
            log_container = self.query_one("#terminal-log", Vertical)
            self._active_prompt_widget = Static(Text.from_markup(self._prompt_text), classes="terminal-entry")
            log_container.mount(self._active_prompt_widget)
            self.call_after_refresh(lambda: self._active_prompt_widget.scroll_visible(animate=False))
        except Exception:
            pass

    def _add_log_entry(self, content) -> None:
        """
        adds a new static entry to the terminal log container.

        args:
            content: the text or text object to display.

        returns:
            none
        """
        try:
            log_container = self.query_one("#terminal-log", Vertical)
            
            if isinstance(content, str):
                text_obj = Text.from_markup(content) if "[" in content else Text(content)
            else:
                text_obj = content
            
            new_entry = Static(text_obj, classes="terminal-entry")
            log_container.mount(new_entry)
            self.call_after_refresh(lambda: new_entry.scroll_visible(animate=False))
        except Exception:
            pass

    def action_clear_terminal(self) -> None:
        """
        resets the terminal panel to its initial state.

        returns:
            none
        """
        try:
            self.query_one("#terminal-log", Vertical).remove_children()
            self._prompt_text = self._get_prompt()
            self._show_prompt()
        except Exception:
            pass

    def poll_terminal(self) -> None:
        """
        polls the reporter for new terminal output and updates the ui.

        returns:
            none
        """
        try:
            reporter = getattr(self.app, "reporter", None)
            if not reporter: return
                
            queue = reporter.terminal_queue
            if not queue: return

            while not queue.empty():
                item = queue.get_nowait()
                try:
                    if not isinstance(item, dict):
                        # simple string updates - if we have an active prompt, maybe combine?
                        # for now, just add a log entry
                        self._add_log_entry(Text.from_ansi(str(item)))
                        continue

                    # handle command dictionary
                    if "command" in item:
                        cmd = str(item["command"])
                        if self._active_prompt_widget:
                            # update the existing prompt with the command text
                            new_text = Text.from_markup(self._prompt_text) + Text(cmd)
                            self._active_prompt_widget.update(new_text)
                            self._active_prompt_widget = None # consumption
                        else:
                            self._add_log_entry(cmd)

                    # handle output dictionary
                    if "output" in item:
                        out_text = str(item["output"])
                        # space before output
                        self._add_log_entry("") 
                        for line in out_text.splitlines():
                            self._add_log_entry(Text.from_ansi(line))
                        # space after output
                        self._add_log_entry("") 

                    # handle new prompt
                    if "new_prompt" in item:
                        self._prompt_text = str(item["new_prompt"])
                        self._show_prompt()

                except Exception as e:
                    self._add_log_entry(Text(f"[UI ERROR] {e}", style="bold red"))
        except Exception:
            pass


# ------------------------------------------------------------------
#  JSON EDITOR
# ------------------------------------------------------------------

class JsonEditor(Widget):
    """
    full-panel json editor overlay for system configuration files.
    """

    DEFAULT_CSS = """
    JsonEditor {
        height: 1fr;
        background: #080808;
        overflow: hidden hidden;
    }
    #editor-scroll {
        height: 1fr;
        padding: 1;
        scrollbar-size: 0 0;
    }
    .file-label {
        color: #f0f0f0;
        padding: 1 1;
        width: auto;
        background: #080808;
        margin-top: 1;
    }
    TextArea {
        height: 15;
        margin-bottom: 1;
        border: solid #2a2a2a;
        scrollbar-size: 0 0;
        background: #080808;
    }
    TextArea:focus {
        border: solid #555555;
    }
    .editor-sys-note {
        margin: 1 0;
        color: #a0a0a0;
    }
    """

    FILES = [
        Path(__file__).parent / "workflow/config/machine_state.json",
        Path(__file__).parent / "workflow/config/archive.json"
    ]

    def __init__(self, **kwargs):
        """
        initializes the json editor widget.

        args:
            **kwargs: keyword arguments for widget.

        returns:
            none
        """
        super().__init__(**kwargs)
        self.areas: dict[Path, TextArea] = {}

    def compose(self) -> ComposeResult:
        """
        composes the editor layout.

        returns:
            ComposeResult: the composed widgets.
        """
        with ScrollableContainer(id="editor-scroll"):
            for path in self.FILES:
                yield Static(f" {path.name}", classes="file-label")
                area = EditorTextArea(language="json")
                self.areas[path] = area
                yield area
            
            yield Static(
                "Ctrl+S to save all & sync   Esc to return to chat",
                classes="editor-sys-note",
            )

    def on_mount(self) -> None:
        """
        loads file contents into text areas when mounted.

        returns:
            none
        """
        for path, area in self.areas.items():
            if path.exists():
                area.load_text(path.read_text())

    def action_save(self) -> None:
        """
        validates and saves contents of all text areas to disk.

        returns:
            none
        """
        saved_files = []
        for path, area in self.areas.items():
            try:
                orjson.loads(area.text)
            except Exception as e:
                if hasattr(self.parent, "_sys"):
                    self.parent._sys(f"SYNTAX ERROR in {path.name}: {e}")
                    self.parent._divider()
                return

            path.write_text(area.text)
            saved_files.append(path.name)

        if hasattr(self.parent, "_sys"):
            files_str = ", ".join(saved_files)
            self.parent._sys(f"Config synchronized: {files_str} updated.")

        self.dismiss_editor()

    def action_exit_editor(self) -> None:
        """
        aborts the editor session without saving.

        returns:
            none
        """
        if hasattr(self.parent, "_sys"):
            self.parent._sys("Editor session aborted — no changes saved.")
        self.dismiss_editor()

    def dismiss_editor(self) -> None:
        """
        removes the editor overlay and restores focus.

        returns:
            none
        """
        try:
            log = self.parent.query_one("#chat-log")
            log.display = True
            self.remove()
            self.parent.query_one("#chat-input").focus()
        except Exception:
            self.remove()


# ------------------------------------------------------------------
#  CONFIG EDITOR (YAML)
# ------------------------------------------------------------------

class ConfigEditor(Widget):
    """
    full-panel editor for sys_config.yaml.
    """

    DEFAULT_CSS = """
    ConfigEditor {
        height: 1fr;
        background: #080808;
        overflow: hidden hidden;
    }
    #editor-scroll {
        height: 1fr;
        padding: 1;
        scrollbar-size: 0 0;
    }
    .file-label {
        color: #f0f0f0;
        padding: 1 1;
        width: auto;
        background: #080808;
    }
    TextArea {
        height: 1fr;
        margin-bottom: 1;
        border: solid #2a2a2a;
        scrollbar-size: 0 0;
        background: #080808;
    }
    TextArea:focus {
        border: solid #555555;
    }
    .editor-sys-note {
        margin: 1 0;
        color: #a0a0a0;
    }
    """

    FILE = Path(__file__).parent / "workflow/config/sys_config.yaml"

    def __init__(self, **kwargs):
        """
        initializes the config editor widget.

        args:
            **kwargs: keyword arguments for widget.

        returns:
            none
        """
        super().__init__(**kwargs)
        self.areas: dict[Path, TextArea] = {}

    def compose(self) -> ComposeResult:
        """
        composes the config editor layout.

        returns:
            ComposeResult: the composed widgets.
        """
        with ScrollableContainer(id="editor-scroll"):
            yield Static(f" {self.FILE.name}", classes="file-label")
            area = EditorTextArea(language="yaml")
            self.areas[self.FILE] = area
            yield area
            yield Static(
                "Ctrl+S to save & sync   Esc to return to chat",
                classes="editor-sys-note",
            )

    def on_mount(self) -> None:
        """
        loads yaml content when mounted.

        returns:
            none
        """
        if self.FILE.exists():
            self.areas[self.FILE].load_text(self.FILE.read_text())

    def action_save(self) -> None:
        """
        saves the yaml content to disk.

        returns:
            none
        """
        if not self.areas or self.FILE not in self.areas:
            return
        
        area = self.areas[self.FILE]
        # we could add yaml validation here if needed
        self.FILE.write_text(area.text)

        if hasattr(self.parent, "_sys"):
            self.parent._sys(f"Config synchronized: {self.FILE.name} updated.")

        self.dismiss_editor()

    def action_exit_editor(self) -> None:
        """
        aborts the session without saving.

        returns:
            none
        """
        if hasattr(self.parent, "_sys"):
            self.parent._sys("Editor session aborted — no changes saved.")
        self.dismiss_editor()

    def dismiss_editor(self) -> None:
        """
        removes the editor and restores focus.

        returns:
            none
        """
        try:
            log = self.parent.query_one("#chat-log")
            log.display = True
            self.remove()
            self.parent.query_one("#chat-input").focus()
        except Exception:
            self.remove()


# ------------------------------------------------------------------
#  AGENT STATUS WIDGET
# ------------------------------------------------------------------

class AgentStatus(Static):
    """
    widget to display live task status and tool execution from the agent.
    """
    DEFAULT_CSS = """
    AgentStatus {
        width: 100%;
        height: auto;
        padding: 0 2;
        margin: 0;
        color: #666666;
        background: #080808;
        border-left: thick #222222;
        overflow-x: hidden;
        display: none;
    }
    """
    STATE_FILE = Path(__file__).parent / "workflow/config/machine_state.json"

    def __init__(self, session_id: str, **kwargs):
        """
        initializes the agent status widget.

        args:
            session_id (str): identifier for the current session.
            **kwargs: keyword arguments.

        returns:
            none
        """
        super().__init__(**kwargs)
        self.session_id = session_id
        self.task_queue = None
        self.task_lines = []
        self.tool_calls = [] 
        self.active = True
        self._polling_task = None
        self.orchestrator_active = False 
        
    def on_mount(self) -> None:
        """
        subscribes to task updates and starts polling.

        returns:
            none
        """
        from supervisor import Reporter
        self.task_queue = Reporter.subscribe_tasks()
        self._polling_task = asyncio.create_task(self._poll_tasks())

    async def _poll_tasks(self) -> None:
        """
        listens for broadcasted task state updates from the supervisor.

        returns:
            none
        """
        status_map = {
            "pending": "[ ]",
            "skipped": "[x]",
            "done":    "[*]",
            "running": "[.]",
        }
        
        # we no longer check the state file immediately to prevent ghosting.
        # we wait for the live broadcast from the supervisor.

        if not self.task_queue:
            return

        while self.active:
            try:
                tasks = await self.task_queue.get()
                self._update_task_lines(tasks, status_map)
                self.task_queue.task_done()
            except Exception:
                await asyncio.sleep(0.1)

    def _update_task_lines(self, tasks: dict, status_map: dict) -> None:
        """
        updates the internal representation of task lines.

        args:
            tasks (dict): current task states.
            status_map (dict): mapping of status names to symbols.

        returns:
            none
        """
        if not tasks: return
        new_lines = []
        # preserve original order from dict
        for tid, t in tasks.items():
            status_sym = status_map.get(t.get("status"), "[?]")
            desc = t.get("description", "Unknown task").replace("\n", " ").strip()
            # clip task description by length 80
            if len(desc) > 80:
                desc = desc[:77] + "..."
            line = f"{status_sym} {desc}"
            new_lines.append(line)
        
        if self.task_lines != new_lines:
            self.task_lines = new_lines
            self.display = True 
            self.styles.margin = (1, 0)
            self.refresh()

    def update_tool(self, name: str, args: dict) -> None:
        """
        adds a new tool execution entry to the status log.

        args:
            name (str): name of the tool being called.
            args (dict): arguments passed to the tool.

        returns:
            none
        """
        n = name.lower()
        if n == "orchestrator":
            self.orchestrator_active = True

        mapping = {
            "web_search": ["searching", "scanning web"],
            "orchestrator": ["planning", "strategizing"],
            "terminal": ["executing", "running command"],
            "memory_query": ["querying memory", "searching logs"],
            "read_file": ["reading"],
            "write_file": ["writing"],
            "replace": ["editing"]
        }
        
        phrases = mapping.get(n, ["calling"])
        phrase = random.choice(phrases)
        
        arg_val = ""
        if n == "web_search": arg_val = args.get("query", "")
        elif n == "terminal": arg_val = args.get("command", "")
        elif n == "orchestrator": arg_val = args.get("task", "")
        elif n == "memory_query": arg_val = args.get("query", "")
        elif "path" in args: arg_val = args.get("path", "")
        elif "file_path" in args: arg_val = args.get("file_path", "")
        elif "command" in args: arg_val = args.get("command", "")
        elif "query" in args: arg_val = args.get("query", "")
        else:
            arg_val = str(list(args.values())[0]) if args else ""

        arg_val = str(arg_val).replace("\n", " ").strip()
        line = f"▶ {phrase}: {arg_val}"
        if len(line) > 75:
            line = line[:72] + "..."
            
        if line not in self.tool_calls:
            self.tool_calls.append(line)
            self.display = True 
            self.styles.margin = (1, 0)
            self.refresh()

    def stop(self) -> None:
        """
        stops polling and unsubscribes from updates.

        returns:
            none
        """
        self.active = False
        if self._polling_task:
            self._polling_task.cancel()
        
        from supervisor import Reporter
        if self.task_queue:
            Reporter.unsubscribe_tasks(self.task_queue)

    def render(self) -> Text:
        """
        renders the combined task and tool status log.

        returns:
            Text: the formatted status text.
        """
        # hide the widget entirely until tools or tasks are actually active.
        if not self.tool_calls and not self.task_lines and not self.orchestrator_active:
            return Text("")
            
        res = Text()
        # add tool calls in a very dimmed style
        for tc in self.tool_calls:
            res.append(f" {tc.lower()}\n", style="#444444")
        
        # show planning indicator if orchestrated but no tasks yet
        if self.orchestrator_active and not self.task_lines:
            label = "Planning workflow..." if self.active else "Planned."
            res.append(f" ⬡ {label}\n", style="#555555")

        # add tasks with requested mapping and highlighting
        if self.task_lines:
            res.append("\n") # space before tasks
            if self.tool_calls:
                res.append(" " + "─" * 40 + "\n", style="#1a1a1a")
            for i, tl in enumerate(self.task_lines):
                style = "#666666" # pending
                if "[*]" in tl: style = "#999999" # done
                elif "[.]" in tl:
                    style = "#ffffff" if self.active else "#aaaaaa"
                elif "[x]" in tl: style = "#cc4444" # skipped/error
                
                res.append(f" {tl}\n", style=style)

                # add vertical separator between tasks
                if i < len(self.task_lines) - 1:
                    res.append("  |\n", style="#1a1a1a")
        
        if not self.active and (self.tool_calls or self.task_lines):
             res.append("\n")
                
        return res





# ------------------------------------------------------------------
#  CHAT MESSAGE
# ------------------------------------------------------------------

class ChatMessage(Static):
    """
    widget to display a single chat message with a role label and timestamp.
    """
    DEFAULT_CSS = """
    ChatMessage {
        width: 100%;
        padding: 0 2;
        margin: 0;
        overflow: hidden hidden;
    }
    ChatMessage.user {
        background: #0d0d0d;
        border-left: thick #555555;
        color: #d8d8d8;
    }
    ChatMessage.assistant {
        background: #0d0d0d;
        border-left: thick #e0e0e0;
        color: #f0f0f0;
    }
    ChatMessage.system {
        background: #0a0a0a;
        border-left: thick #333333;
        color: #999999;
    }
    ChatMessage.divider {
        background: #060606;
        border-left: thick #060606;
        color: #1a1a1a;
        padding: 0 2;
    }
    """

    def __init__(self, role: str, content: str, timestamp: str, **kwargs):
        """
        initializes a chat message.

        args:
            role (str): message sender role (user, assistant, system, divider).
            content (str): textual content of the message.
            timestamp (str): timestamp string.
            **kwargs: keyword arguments.

        returns:
            none
        """
        super().__init__(**kwargs)
        self.role      = role
        self.content   = content
        self.timestamp = timestamp
        self.add_class(role)

    def update_content(self, content: str) -> None:
        """
        updates the message content and refreshes the widget.

        args:
            content (str): new content.

        returns:
            none
        """
        self.content = content
        self.refresh()

    def render(self) -> Text:
        """
        renders the message with role-specific styling and escaping.

        returns:
            Text: rich text representation.
        """
        if self.role == "divider":
            return Text("─" * 60, style="#1e1e1e")

        labels = {
            "user":      ("▶  You",    "#888888"),
            "assistant": ("◆  Eye",    "#ffffff"),
            "system":    ("⊙  System", "#555555"),
        }
        label, color = labels.get(self.role, ("?", "white"))

        t = Text()
        t.append(label,  style=f"bold {color}")
        t.append(f"   {self.timestamp}\n", style="dim")
        
        # support rich markup in content (mostly for system/session messages)
        # escape content for user/assistant to prevent accidental markup injection
        display_content = self.content
        if self.role != "system":
            display_content = escape(display_content)
            
        try:
            t.append(Text.from_markup(display_content))
        except Exception:
            t.append(self.content)
        return t


# ------------------------------------------------------------------
#  WELCOME SCREEN
# ------------------------------------------------------------------

class WelcomeScreen(Static):
    """
    static splash screen shown at startup.
    """
    DEFAULT_CSS = """
    WelcomeScreen {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: #3a3a3a;
        background: #080808;
    }
    """

    def render(self) -> Text:
        """
        renders the welcome banner.

        returns:
            Text: rich representation of the welcome graphic.
        """
        return Text(WELCOME, style="#3a3a3a")


# ------------------------------------------------------------------
#  CHAT PANEL
# ------------------------------------------------------------------

class ChatPanel(Widget):
    """
    the primary interactive chat panel.
    """
    DEFAULT_CSS = """
    ChatPanel {
        width: 1fr;
        height: 1fr;
        background: #080808;
        layout: vertical;
        overflow-x: hidden;
    }

    #chat-titlebar {
        height: 3;
        background: #050505;
        border-bottom: double #2a2a2a;
        content-align: center middle;
        padding: 0 2;
        align: center middle;
    }
    #chat-titlebar-label {
        width: 1fr;
        height: 3;
        content-align: center middle;
        color: #aaaaaa;
    }

    #welcome-body {
        height: 1fr;
    }

    #chat-log {
        height: 1fr;
        overflow-y: auto;
        padding: 0 0 0 0;
        background: #080808;
        scrollbar-size: 0 0;
    }
    #chat-log > *:first-child {
        margin-top: 1;
    }

    #chat-input {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 3;
        border: none;
        background: rgba(0,0,0,0);
        color: #cccccc;
        scrollbar-size: 0 0;
        scrollbar-gutter: stable;
    }

    #input-row {
        height: auto;
        min-height: 3;
        max-height: 5;
        border: solid #1a1a1a;
        background: rgba(0,0,0,0);
        padding: 0;
        layout: horizontal;
        align: left middle;
        scrollbar-size: 0 0;
    }
    #chat-input:focus {
        background: rgba(0,0,0,0);
        border: none;
    }

    #chat-statusbar {
        height: 2;
        padding: 0 1;
        color: #2e2e2e;
        content-align: left middle;
        background: #050505;
    }

    """

    message_count: reactive[int] = reactive(0)
    _welcomed: bool = False
    _exchange_count: int = 0
    _awaiting_confirm: bool = False

    def __init__(self, **kwargs):
        """
        initializes the chat panel.

        args:
            **kwargs: keyword arguments.

        returns:
            none
        """
        super().__init__(**kwargs)
        # reporter is now managed by the app
        self.session_id = self._next_session_id()
        self.current_worker = None

    @property
    def reporter(self) -> Reporter:
        """
        retrieves the global reporter instance from the app.

        returns:
            Reporter: the shared reporter agent.
        """
        return self.app.reporter

    def _next_session_id(self) -> str:
        """
        generates the next available numeric session id between 1 and 10.

        returns:
            str: the next session id.
        """
        archive = self.reporter.memory.get_archive()
        # find all numeric keys in archive
        numeric_ids = {int(sid) for sid in archive.keys() if sid.isdigit()}
        
        # try to find the first free id in 1-10
        for i in range(1, 11):
            if i not in numeric_ids:
                return str(i)
        
        # if all 1-10 are taken, find the oldest one (the first key in insertion-ordered dict)
        if archive:
            oldest_id = next(iter(archive))
            return str(oldest_id)
            
        return "1"

    def _get_active_sessions(self) -> list[str]:
        """
        returns a sorted list of numeric session ids.

        returns:
            list[str]: list of session identifiers.
        """
        archive = self.reporter.memory.get_archive()
        numeric_ids = sorted([int(sid) for sid in archive.keys() if sid.isdigit()])
        return [str(sid) for sid in numeric_ids]

    def compose(self) -> ComposeResult:
        """
        composes the chat layout.

        returns:
            ComposeResult: composed widgets.
        """
        yield WelcomeScreen(id="welcome-body")
        with Horizontal(id="input-row"):
            yield HistoryInput(placeholder="  >_", id="chat-input")
        yield Static(
            " ⬡ Initializing...",
            id="chat-statusbar",
        )

    def on_mount(self) -> None:
        """
        starts startup checks and initializes memory session.

        returns:
            none
        """
        self.run_worker(self._startup_check())
        self.reporter.memory.create_session(self.session_id)

    async def _startup_check(self) -> None:
        """
        perform system checks at startup and update status bar.

        returns:
            none
        """
        agent_online = False
        shell_active = False
        crawler_online = False
        
        try:
            # 1. check agent (llm) connectivity
            try:
                agent_res = await self.reporter.ping()
                if agent_res:
                    agent_online = True
            except Exception:
                agent_online = False

            # 2. check shell and crawler (mcp server) connectivity
            try:
                ping_data = await self.reporter.mcp_client.ping()
                if isinstance(ping_data, dict):
                    shell_active = ping_data.get("status") == "success"
                    crawler_online = ping_data.get("crawler") == "online"
            except Exception:
                shell_active = False
                crawler_online = False

            # 3. update status bar
            agent_status = "ONLINE" if agent_online else "OFFLINE"
            shell_status = "ACTIVE" if shell_active else "INACTIVE"
            crawler_status = "ONLINE" if crawler_online else "OFFLINE"
            
            # dynamic readiness indicator
            is_ready = agent_online and shell_active and crawler_online
            ready_label = "READY" if is_ready else "NOT READY"
            
            status_text = f" ⬡ {ready_label}  ·  AGENT: {agent_status}  ·  SHELL: {shell_status}  ·  CRAWLER: {crawler_status}"
            self.query_one("#chat-statusbar", Static).update(status_text)
            
        except Exception:
            try:
                self.query_one("#chat-statusbar", Static).update(" ⬡ NOT READY  ·  AGENT: OFFLINE  ·  SHELL: INACTIVE  ·  CRAWLER: OFFLINE")
            except Exception:
                pass

    def _ts(self) -> str:
        """
        generates current timestamp.

        returns:
            str: formatted timestamp.
        """
        return datetime.now().strftime("%H:%M:%S")

    def _ensure_log(self) -> ScrollableContainer:
        """
        ensures the chat log container exists, removing welcome screen if needed.

        returns:
            ScrollableContainer: the chat log container.
        """
        if not self._welcomed:
            self._welcomed = True
            self.query_one("#welcome-body").remove()
            log = ScrollableContainer(id="chat-log")
            self.mount(log, before=self.query_one("#input-row"))
        return self.query_one("#chat-log", ScrollableContainer)

    _last_role = None

    def _post(self, role: str, content: str) -> ChatMessage:
        """
        posts a new message to the chat log.

        args:
            role (str): sender role.
            content (str): message text.

        returns:
            ChatMessage: the created message widget.
        """
        log = self._ensure_log()
        
        # treat status and assistant as part of the same assistant turn
        is_assistant_group = role in ("assistant", "status")
        last_was_assistant_group = self._last_role in ("assistant", "status")

        if self._last_role is not None and self._last_role != role:
            # skip divider if switching within assistant group or from user to assistant
            # (since we now add a divider at the end of every assistant response)
            is_u_to_a = (role in ("assistant", "status") and self._last_role == "user")
            if not (is_assistant_group and last_was_assistant_group) and role != "divider" and not is_u_to_a:
                self._smart_divider()
            
        self._last_role = role
        msg = ChatMessage(role=role, content=content, timestamp=self._ts())
        log.mount(msg)
        self.message_count += 1
        self.call_after_refresh(log.scroll_end, animate=True)
        return msg

    def _divider(self) -> None:
        """
        adds a visual divider to the chat log.

        returns:
            none
        """
        log = self._ensure_log()
        log.mount(ChatMessage(role="divider", content="", timestamp=""))
        self._last_role = "divider"
        self.call_after_refresh(log.scroll_end, animate=True)

    def _smart_divider(self) -> None:
        """
        adds a divider only if the last entry wasn't already a divider.

        returns:
            none
        """
        log = self._ensure_log()
        if not log.children: return
        last = log.children[-1]
        if isinstance(last, ChatMessage) and last.role == "divider":
            return
        self._divider()

    def _sys(self, msg: str) -> None: 
        """
        shortcut to post a system message.
        """
        self._post("system", msg)
        
    def _you(self, msg: str) -> None: 
        """
        shortcut to post a user message.
        """
        self._post("user", msg)
        
    def _eye(self, msg: str) -> ChatMessage: 
        """
        shortcut to post an assistant message.
        """
        return self._post("assistant", msg)

    async def _get_reporter_response(self, text: str) -> None:
        """
        handles the async streaming response from the reporter agent.

        args:
            text (str): user input query.

        returns:
            none
        """
        log = self._ensure_log()
        full_response = ""
        msg_widget = None
        
        # mount status immediately but it will render nothing until a tool is called
        self._last_role = "status"

        status_widget = AgentStatus(self.session_id)
        log.mount(status_widget)
        self.call_after_refresh(log.scroll_end, animate=True)

        try:
            async for chunk in self.reporter(text, self.session_id):
                # handle tool calls
                if isinstance(chunk, dict) and "tool" in chunk:
                    status_widget.update_tool(chunk["tool"], chunk.get("args", {}))
                    continue

                # handle text chunks
                if isinstance(chunk, str):
                    if not msg_widget:
                        msg_widget = self._eye("")
                        # scroll once when the response actually begins
                        self.call_after_refresh(log.scroll_end, animate=True)
                    
                    full_response += chunk
                    msg_widget.update_content(full_response)

        except Exception as e:
            self._sys(f"REPORTER ERROR: {e}")
            if status_widget:
                status_widget.remove()
            self.current_worker = None
            return 
        finally:
            if status_widget:
                try:
                    status_widget.stop()
                except Exception:
                    pass
            
            # ensure the log scrolls to the bottom
            self._divider()
            self.call_after_refresh(log.scroll_end, animate=True)
            self.current_worker = None

    def _dismiss_any_editor(self) -> bool:
        """
        dismiss any open editor without saving.
    
        returns:
            bool: true if an editor was found and closed.
        """
        dismissed = False
        for editor_type in (ConfigEditor, JsonEditor):
            try:
                editor = self.query_one(editor_type)
                editor.remove()
                dismissed = True
            except Exception:
                pass

        if dismissed:
            try:
                log = self.query_one("#chat-log")
                log.display = True
            except Exception:
                pass

        return dismissed

    def _open_config_editor(self):
        """
        opens the yaml config editor overlay.

        returns:
            none
        """
        self._dismiss_any_editor()

        log = self._ensure_log()
        log.display = False

        editor = ConfigEditor()
        self.mount(editor, before="#input-row")

        def focus_init():
            if editor.areas:
                list(editor.areas.values())[0].focus()

        self.call_after_refresh(focus_init)

    def _open_json_editor(self):
        """
        opens the json history/state editor overlay.

        returns:
            none
        """
        self._dismiss_any_editor()

        log = self._ensure_log()
        log.display = False

        editor = JsonEditor()
        self.mount(editor, before="#input-row")

        def focus_init():
            if editor.areas:
                list(editor.areas.values())[0].focus()

        self.call_after_refresh(focus_init)

    def send(self) -> None:
        """
        processes the current input text and triggers agent execution.

        returns:
            none
        """
        inp = self.query_one("#chat-input", HistoryInput)
        text = inp.text.strip()
        if not text:
            return

        # preemption logic: if a worker is running, cancel it
        if self.current_worker and self.current_worker.is_running:
            self.current_worker.cancel()
            self._sys("Process interrupted.")

        for editor_type in (ConfigEditor, JsonEditor):
            try:
                editor = self.query_one(editor_type)
                editor.action_exit_editor() 
            except Exception:
                pass

        inp.text = ""
        inp.push_history(text)

        if text.startswith("/"):
            self._handle_command(text)
            return

        self._you(text)
        self._divider()
        self._exchange_count += 1

        # run the reporter agent in a background worker
        self.current_worker = self.run_worker(self._get_reporter_response(text))
        
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        event handler for chat input submission.

        args:
            event (Input.Submitted): the submission event.

        returns:
            none
        """
        self.send()

    def on_static_click(self, event: events.Click) -> None:
        """
        event handler for click events on static widgets.

        args:
            event (events.Click): the click event.

        returns:
            none
        """
        if event.widget.id == "send-btn":
            self.send()

    def _handle_command(self, raw: str) -> None:
        """
        parses and executes slash commands.

        args:
            raw (str): the raw command string.

        returns:
            none
        """
        cmd = raw.lower().strip()

        if cmd == "/help":
            self._sys(
                "Available commands:\n"
                "\n"
                "  /help             — Show this help\n"
                "  /new              — Start a new empty session\n"
                "  /session          — List all available sessions\n"
                "  /session <id>     — Switch to an existing session\n"
                "  /clear            — Purge current session state\n"
                "  /purge            — Global reset (WIPE ALL SESSIONS)\n"
                "  /config           — Edit machine_state.json\n"
                "  /edit             — Edit sys_config.yaml\n"
                "  /quit             — Exit God's Eye\n"
            )

        elif cmd == "/new":
            self.session_id = self._next_session_id()
            self.reporter.memory.create_session(self.session_id)
            if self._welcomed:
                self.query_one("#chat-log", ScrollableContainer).remove_children()
            self._exchange_count = 0
            self._sys(f"Started new session: {self.session_id}")

        elif cmd == "/clear":
            # purge current session only
            self.reporter.clear_session(self.session_id)
            if self._welcomed:
                self.query_one("#chat-log", ScrollableContainer).remove_children()
            
            # also clear the terminal panel
            self.app.query_one(LeftPanel).action_clear_terminal()
            
            self.message_count = 0
            self._sys(f"Session '{self.session_id}' state purged. Memory and history cleared for this session.")

        elif cmd == "/purge":
            # global purge
            self.reporter.purge_all()
            self.session_id = "1"
            if self._welcomed:
                self.query_one("#chat-log", ScrollableContainer).remove_children()
            self._exchange_count = 0
            self._sys("GLOBAL PURGE COMPLETE. All sessions, memory, and history have been wiped.")

        elif cmd.startswith("/session"):
            parts = cmd.split()
            sessions = self._get_active_sessions()

            if len(parts) == 1:
                # list sessions
                if not sessions:
                    self._sys("No active sessions found.")
                else:
                    lines = []
                    for s in sessions:
                        is_current = (s == self.session_id)
                        if is_current:
                            lines.append(f" [bold white on #333333]  • {s}  [/]")
                        else:
                            lines.append(f"   • {s}")
                    
                    self._sys("Available sessions:\n" + "\n".join(lines))
                    self._sys("\nType '/session <id>' to switch.")
            else:
                # switch session
                target = parts[1]
                
                # verify session existence
                if target not in sessions:
                    self._sys(f"Session '{target}' not found.")
                    return

                if target == self.session_id:
                    self._sys(f"Already in session '{target}'.")
                else:
                    self.session_id = target
                    # reload chat log from archive
                    archive = self.reporter.memory.get_archive()
                    log = self._ensure_log()
                    log.remove_children()

                    session_messages = archive.get(target, [])
                    for msg in session_messages:
                        # re-render each historical message
                        role = msg.get("role", "system")
                        content = msg.get("content", "")
                        self._post(role, content)
                        self._divider()

                    self._sys(f"switched to session '{target}'.")

        elif cmd == "/config":
            self._open_json_editor()

        elif cmd == "/quit":
            self.app.exit()

        elif cmd == "/edit":
            self._open_config_editor()

        else:
            self._sys(f"Unknown command: {raw}\nType /help for available commands.\n")


# ------------------------------------------------------------------
#  MAIN APP
# ------------------------------------------------------------------

class GodsEye(App):
    """
    main terminal application class.
    """
    TITLE     = "God's Eye"
    SUB_TITLE = "Terminal intelligence interface"

    CSS = """
    Screen {
        layout: vertical;
        background: #050505;
    }

    #top-header {
        height: 1;
        background: #1a1a1a;
        padding: 0 2;
        layout: horizontal;
        color: #e8e8e8;
    }
    #top-header-left  { width: 1fr;  content-align: left middle; }
    #top-header-right { width: auto; content-align: right middle; }

    #main-body {
        height: 1fr;
        layout: horizontal;
    }

    #footer-keys  { width: 1fr;  content-align: left middle; }
    #footer-state { width: auto; content-align: right middle; color: #555555; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit",       "Quit",  show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("ctrl+h", "show_help",  "Help",  show=True),
        Binding("ctrl+s", "save",       "Save"),
    ]

    def __init__(self, **kwargs):
        """
        initializes the application and the reporter agent.

        args:
            **kwargs: keyword arguments.

        returns:
            none
        """
        super().__init__(**kwargs)
        self.reporter = Reporter()

    def compose(self) -> ComposeResult:
        """
        composes the main application layout.

        returns:
            ComposeResult: composed widgets.
        """
        with Horizontal(id="top-header"):
            yield Static("[bold]God's Eye ⚆ [/]", id="top-header-left")
            yield Static(datetime.now().strftime("  [bold]%Y-%m-%d[/]  "), id="top-header-right")
        with Horizontal(id="main-body"):
            yield LeftPanel(id="left-panel")
            yield ChatPanel(id="right-panel")

    def on_mount(self) -> None:
        """
        focuses the chat input on startup.

        returns:
            none
        """
        self.query_one("#chat-input", HistoryInput).focus()

    def action_quit(self) -> None:
        """
        exits the application.

        returns:
            none
        """
        self.exit()

    def action_clear_chat(self) -> None:
        """
        triggers the /clear command.

        returns:
            none
        """
        self.query_one(ChatPanel)._handle_command("/clear")

    def action_show_help(self) -> None:
        """
        triggers the /help command.

        returns:
            none
        """
        self.query_one(ChatPanel)._handle_command("/help")


if __name__ == "__main__":
    eye = GodsEye()
    eye.run()