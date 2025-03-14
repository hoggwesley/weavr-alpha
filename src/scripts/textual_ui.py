import subprocess
import json
from rich.text import Text  # ✅ Fix for MarkupError
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Markdown, Static
from textual.containers import Container
from textual.reactive import reactive
import os
import sys

# ✅ Ensure Python finds 'modules/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

class WeavrTUI(App):
    """Textual-based TUI for Weavr AI"""
    BINDINGS = [
        ("r", "toggle_rag", "Toggle RAG"),
        ("q", "quit", "Quit Weavr AI"),
        ("escape", "quit", "Force Quit"),  # ✅ Press ESC to exit immediately
    ]

    # ✅ Reactive state
    current_model = reactive("Loading...")
    use_rag = reactive(False)

    def compose(self) -> ComposeResult:
        """Build the UI components."""
        yield Header()
        yield Container(
            Markdown("## Weavr AI TUI\nType a query below and press Enter."),
            Input(placeholder="Ask Weavr AI something..."),
            Static("", id="response"),  # AI Response area
        )
        yield Footer()

    async def on_mount(self) -> None:
        """Runs when the app starts."""
        self.query_one(Input).focus()

        try:
            # ✅ Get the correct Python path from the venv
            venv_python = os.path.join(sys.prefix, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(sys.prefix, "bin", "python")

            self.process = subprocess.Popen(
                [venv_python, "src/scripts/run_weavr.py"],  # ✅ Use the Python from the venv
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
)

            # ✅ Check if the subprocess started successfully
            if self.process.poll() is not None:
                error_msg = self.process.stderr.read().strip()
                self.query_one("#response", Static).update(
                    Text(f"❌ Error: `run_weavr.py` failed to start.\n{error_msg}", no_wrap=True)
                )
                self.process = None
        except Exception as e:
            self.query_one("#response", Static).update(
                Text(f"❌ Error starting `run_weavr.py`: {str(e)}", no_wrap=True)
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handles user queries when they press Enter."""
        query = event.value.strip()

        if not query or not self.process or not self.process.stdin:
            self.query_one("#response", Static).update(Text("❌ Error: Weavr AI is not running.", no_wrap=True))
            return

        # ✅ Prevent `[Errno 22] Invalid Argument`
        if not query.isprintable():
            self.query_one("#response", Static).update(Text("❌ Invalid input: Non-printable character detected.", no_wrap=True))
            return

        self.query_one(Input).clear()

        try:
            # ✅ Send query to run_weavr.py
            if self.process.poll() is not None:  # ✅ Check if Weavr AI is still running
                self.query_one("#response", Static).update(Text("❌ Error: Weavr AI process has stopped.", no_wrap=True))
                return

            if self.process.stdin and not self.process.stdin.closed:  # ✅ Prevent `[Errno 22]`
                self.process.stdin.write(query + "\n")
                self.process.stdin.flush()
            else:
                self.query_one("#response", Static).update(Text("❌ Error: Weavr AI input is closed.", no_wrap=True))

            # ✅ Read and process the response
            response_json = self.process.stdout.readline().strip()
            response = json.loads(response_json)

            if response["type"] == "response":
                self.query_one("#response", Static).update(
                    Text(f"### AI Response:\n{response['text']}\n🔹 Tokens used: {response['tokens']}", no_wrap=True)
                )
        except Exception as e:
            error_text = Text(f"❌ Error sending query: {str(e)}", no_wrap=True)
            self.query_one("#response", Static).update(error_text)

    async def action_quit(self) -> None:
        """Allow user to exit safely with `q` or `ESC`."""
        self.exit()

    async def on_unmount(self) -> None:
        """Cleans up the subprocess on exit."""
        if hasattr(self, "process") and self.process:
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()  # ✅ CLOSE stdin properly
                
                self.process.terminate()
                self.process.wait(timeout=2)  # ✅ Wait for process to fully exit
            except (OSError, ValueError):
                pass  # ✅ Ignore errors when process is already closed

if __name__ == "__main__":
    WeavrTUI().run()
