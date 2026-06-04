"""Regression test for the slash-command autocomplete Enter key (issue #2478).

Typing "/" opens the slash-command menu. Arrow keys move the highlight. Pressing
Enter on a highlighted command must select it (insert the token into the
composer). It used to do nothing useful: a sibling "Enter to send" keydown
listener on the same textarea also fired and submitted the message, because the
autocomplete handler called only e.preventDefault() (which cancels the browser
default, not other JS listeners). The fix adds e.stopImmediatePropagation() to
the insert branch so the autocomplete owns Enter while the menu is open.

The test loads static/js/slashAutocomplete.js in a Node vm sandbox with a small
DOM stub, registers the real autocomplete listener plus a sibling "send"
listener on the same textarea (mirroring app.js), and dispatches Enter.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "static" / "js" / "slashAutocomplete.js"

pytestmark = pytest.mark.skipif(
    not shutil.which("node"), reason="node binary not on PATH"
)

# Harness: load the module source, replace the slashCommands import with a tiny
# fake registry, strip the ESM export, run in a vm sandbox with a DOM stub, then
# wire a sibling "send" listener (like static/app.js) and dispatch keydowns.
HARNESS = r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

let src = fs.readFileSync(process.env.MODULE_PATH, 'utf8');

// Replace the registry import with a deterministic fake. "/note" carries the
// alias "n", which is what made typing "/n" hit the exactHit branch even when a
// different row was highlighted.
src = src.replace(
  /import \{ COMMANDS, LEGACY_ALIASES \} from '\.\/slashCommands\.js';/,
  `const COMMANDS = {
     new:  { handler() {}, category: 'Chat', help: 'New chat' },
     note: { handler() {}, alias: ['n'], category: 'Notes', help: 'Note' },
     doc:  { handler() {}, category: 'Docs', help: 'Document' },
   };
   const LEGACY_ALIASES = {};`
);
src = src.replace(/export function /g, 'function ');
src = src.replace(/export default[^\n]*\n?/g, '');
src += '\nthis.__initSlashAutocomplete = initSlashAutocomplete;';

// --- Minimal DOM stub --------------------------------------------------------
class EventTarget2 {
  constructor() { this._l = {}; }
  addEventListener(t, fn) { (this._l[t] ||= []).push(fn); }
  removeEventListener(t, fn) {
    if (this._l[t]) this._l[t] = this._l[t].filter(f => f !== fn);
  }
  dispatchEvent(ev) {
    ev.target = ev.target || this;
    const list = (this._l[ev.type] || []).slice();
    for (const fn of list) {
      if (ev._stopImmediate) break;
      fn(ev);
    }
    return !ev.defaultPrevented;
  }
}

class El extends EventTarget2 {
  constructor(tag) {
    super();
    this.tagName = (tag || 'div').toUpperCase();
    this.style = {};
    this.children = [];
    this.dataset = {};
    this._value = '';
    this._html = '';
  }
  get value() { return this._value; }
  set value(v) { this._value = v; }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = v; }
  setAttribute() {}
  appendChild(c) { this.children.push(c); return c; }
  focus() {}
  setSelectionRange() {}
  getBoundingClientRect() { return { left: 0, top: 200, right: 300, bottom: 230, width: 300 }; }
  querySelector() { return null; }
  scrollIntoView() {}
}

function makeKeyEvent(key, opts = {}) {
  return {
    type: 'keydown',
    key,
    shiftKey: !!opts.shiftKey,
    defaultPrevented: false,
    _stopImmediate: false,
    preventDefault() { this.defaultPrevented = true; },
    stopPropagation() {},
    stopImmediatePropagation() { this._stopImmediate = true; },
  };
}

const popupStore = {};
const document = new EventTarget2();
document.body = new El('body');
document.getElementById = (id) => popupStore[id] || null;
document.createElement = (tag) => new El(tag);
// _ensurePopup creates the popup via createElement then body.appendChild; capture
// it so getElementById('slash-autocomplete') returns the same node afterwards.
document.createElement = (tag) => {
  const el = new El(tag);
  Object.defineProperty(el, 'id', {
    configurable: true,
    set(v) { this._id = v; if (v) popupStore[v] = this; },
    get() { return this._id; },
  });
  return el;
};

const sandbox = {
  console,
  document,
  window: new EventTarget2(),
  Event: class { constructor(t) { this.type = t; } },
};
sandbox.window.innerHeight = 800;
sandbox.window.innerWidth = 1000;

vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: process.env.MODULE_PATH });

// --- Wire it up like the real app -------------------------------------------
const ta = new El('textarea');
const win = sandbox.window;

// The composer "Enter to send" listener mirrors static/app.js. It is registered
// FIRST (app.js bootstrap), before the autocomplete module loads via dynamic
// import. It defers to the slash menu through the window hook, exactly like the
// real handler does for window._ghostAutocomplete.
let sendCalled = 0;
ta.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    if (win._slashAutocomplete && win._slashAutocomplete.isOpen()) {
      if (win._slashAutocomplete.acceptOnEnter()) {
        e.preventDefault();
        e.stopImmediatePropagation();
        return;
      }
    }
    sendCalled++;
    ta.value = '';  // app.js clears the composer on submit
  }
});

// Autocomplete initializes after the send listener (dynamic import in chat.js).
sandbox.__initSlashAutocomplete(ta);

function openMenu(text) {
  ta.value = text;
  ta.dispatchEvent({
    type: 'input', target: ta, defaultPrevented: false, _stopImmediate: false,
    preventDefault() {}, stopPropagation() {}, stopImmediatePropagation() {},
  });
}

const results = {};

// Case 1: reporter's flow. "/" open, arrow to a row, Enter -> insert, no send.
openMenu('/');
ta.dispatchEvent(makeKeyEvent('ArrowDown'));   // move highlight off row 0
const evInsert = makeKeyEvent('Enter');
ta.dispatchEvent(evInsert);
results.insert_value = ta.value;
results.insert_send_called = sendCalled;
results.insert_default_prevented = evInsert.defaultPrevented;

// Case 2: typed-out exact command "/note" -> let submit pass through.
sendCalled = 0;
openMenu('/note');
const evExact = makeKeyEvent('Enter');
ta.dispatchEvent(evExact);
results.exact_send_called = sendCalled;

console.log(JSON.stringify(results));
"""


def _run():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", HARNESS],
        cwd=ROOT,
        env={"MODULE_PATH": str(MODULE), "PATH": __import__("os").environ["PATH"]},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"node harness failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_enter_inserts_highlighted_command_without_sending():
    r = _run()
    # The highlighted row gets inserted into the composer with a trailing space.
    assert r["insert_value"].strip() != "", "Enter should insert the highlighted command"
    assert r["insert_value"].startswith("/"), r["insert_value"]
    # And the sibling send handler must NOT fire (no message submitted).
    assert r["insert_send_called"] == 0, "Enter must not submit while the menu inserts"
    assert r["insert_default_prevented"] is True


def test_enter_on_exact_typed_command_lets_submit_pass():
    r = _run()
    # When the user has typed a full command, Enter should fall through to submit.
    assert r["exact_send_called"] == 1, "exact command + Enter should submit"
