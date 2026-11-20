"""Offline checks for docs/index.html, the research showcase served by GitHub Pages.

The page is one self-contained file that makes three kinds of promise: that it
embeds nothing (no script, no tracker, no participant recording), that the
numbers it draws agree with each other, and that its links land somewhere.
These tests hold it to those promises by reading files only: no browser and no
network. ``scripts/verify_showcase.cjs`` covers what needs a real browser.

Links into the sibling ProSense repository are checked only when a checkout is
named in ``PROSENSE_ROOT``; otherwise that test is skipped, not passed.
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "docs" / "index.html"

# Outside hosts the page links to today. A new host here should be a decision
# someone made, not a side effect, so the list is spelled out.
KNOWN_HOSTS = {
    "github.com",
    "doi.org",
    "www.thinkmind.org",
    "run.pavlovia.org",
    "gitlab.pavlovia.org",
}

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Node:
    def __init__(self, tag, attrs=None, parent=None):
        self.tag = tag
        self.attrs = dict(attrs or [])
        self.parent = parent
        self.children = []  # Node or str

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def find_all(self, tag=None, **attrs):
        for node in self.walk():
            if tag and node.tag != tag:
                continue
            if all(node.attrs.get(k.rstrip("_")) == v for k, v in attrs.items()):
                yield node

    def text(self):
        parts = []
        for child in self.children:
            parts.append(child if isinstance(child, str) else child.text())
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    def classes(self):
        return (self.attrs.get("class") or "").split()


class TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#document")
        self.current = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.current)
        self.current.children.append(node)
        if tag not in VOID:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.current.children.append(Node(tag, attrs, self.current))

    def handle_endtag(self, tag):
        node = self.current
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self.current = node.parent

    def handle_data(self, data):
        self.current.children.append(data)


@pytest.fixture(scope="module")
def page():
    builder = TreeBuilder()
    builder.feed(PAGE.read_text(encoding="utf-8"))
    builder.close()
    return builder.root


def github_slug(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading (lowercase, punctuation dropped, spaces to hyphens)."""
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def markdown_heading_slugs(path: Path) -> set:
    slugs, in_fence = set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        match = None if in_fence else re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match:
            slugs.add(github_slug(match.group(1)))
    return slugs


def outside_links(page):
    return [a.attrs["href"] for a in page.find_all("a") if a.attrs.get("href", "").startswith(("http://", "https://"))]


def check_repository_links(links, owner_repo: str, root: Path, default_branch: str):
    """Every link into ``owner_repo`` must name a path that exists under ``root``."""
    problems, checked = [], 0
    prefix = f"https://github.com/{owner_repo}"
    for href in links:
        parsed = urlparse(href)
        full = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if full != prefix and not full.startswith(prefix + "/"):
            continue
        checked += 1
        rest = parsed.path[len(f"/{owner_repo}"):].strip("/")
        if rest == "":
            target = root / "README.md"
        else:
            kind, _, tail = rest.partition("/")
            ref, _, rel = tail.partition("/")
            if kind == "tree" and not rel:
                continue  # a link to the tree at a commit: nothing to resolve on disk
            if kind not in {"blob", "tree"}:
                problems.append(f"{href}: unexpected link shape")
                continue
            if not re.fullmatch(r"[0-9a-f]{40}", ref) and ref != default_branch:
                problems.append(f"{href}: branch {ref!r} is not the default branch {default_branch!r}")
            target = root / rel
        if not target.exists():
            problems.append(f"{href}: {target.relative_to(root)} does not exist")
            continue
        if parsed.fragment:
            if target.suffix.lower() != ".md":
                problems.append(f"{href}: anchor on a file that is not Markdown")
            elif parsed.fragment not in markdown_heading_slugs(target):
                problems.append(f"{href}: no heading in {target.relative_to(root)} makes the anchor #{parsed.fragment}")
    return checked, problems


def test_document_basics(page):
    html = next(page.find_all("html"))
    assert html.attrs.get("lang") == "en"
    assert next(page.find_all("title")).text()
    assert len(list(page.find_all("h1"))) == 1
    mains = list(page.find_all("main"))
    assert len(mains) == 1 and mains[0].attrs.get("id") == "main"
    first_link = next(page.find_all("a"))
    assert first_link.attrs.get("href") == "#main", "the skip link must be the first link on the page"


def test_page_embeds_nothing(page):
    """The page says nothing is embedded in it. Keep that literally true."""
    forbidden = {"script", "img", "picture", "svg", "iframe", "frame", "object", "embed", "video", "audio",
                 "source", "track", "form", "input", "button", "select", "textarea", "link", "base"}
    found = sorted({node.tag for node in page.walk() if node.tag in forbidden})
    assert not found, f"embedded or interactive elements appeared: {found}"
    handlers = sorted({key for node in page.walk() for key in node.attrs if key.lower().startswith("on")})
    assert not handlers, f"inline event handlers appeared: {handlers}"
    refresh = [m for m in page.find_all("meta") if (m.attrs.get("http-equiv") or "").lower() == "refresh"]
    assert not refresh, "a meta refresh would send the visitor elsewhere"
    css = " ".join(style.text() for style in page.find_all("style"))
    inline = " ".join(node.attrs.get("style", "") for node in page.walk())
    assert not re.search(r"@import|url\(", css + inline, re.I), "CSS must not load anything"


def test_in_page_links_have_targets_and_ids_are_unique(page):
    ids = [node.attrs["id"] for node in page.walk() if "id" in node.attrs]
    assert len(ids) == len(set(ids)), "duplicate id"
    targets = [a.attrs["href"][1:] for a in page.find_all("a") if a.attrs.get("href", "").startswith("#")]
    assert targets, "the page should have in-page navigation"
    missing = sorted({target for target in targets if target not in ids})
    assert not missing, f"in-page links without a target: {missing}"


def test_outside_links_are_https_and_on_known_hosts(page):
    links = outside_links(page)
    assert links
    assert all(href.startswith("https://") for href in links)
    hosts = {urlparse(href).netloc for href in links}
    assert hosts <= KNOWN_HOSTS, f"new outside host(s) {sorted(hosts - KNOWN_HOSTS)}: add them to KNOWN_HOSTS on purpose"
    for anchor in page.find_all("a"):
        if anchor.attrs.get("target") == "_blank":
            assert "noopener" in (anchor.attrs.get("rel") or "")


def test_bars_agree_with_their_labels_and_the_stated_gap(page):
    bars = [node for node in page.find_all("div", role="img") if "bar" in node.classes()]
    assert len(bars) == 2, "expected the model bar and the baseline bar"
    values = []
    for bar in bars:
        spoken = re.search(r"([\d.]+) percent", bar.attrs["aria-label"]).group(1)
        fill = next(bar.find_all("i"))
        drawn = re.search(r"width:\s*([\d.]+)%", fill.attrs["style"]).group(1)
        siblings = [child for child in bar.parent.children if isinstance(child, Node)]
        label = siblings[siblings.index(bar) - 1]
        assert "measure-label" in label.classes()
        shown = re.search(r"([\d.]+)%", label.text()).group(1)
        assert spoken == drawn == shown, f"bar disagrees with itself: says {shown}, draws {drawn}, announces {spoken}"
        values.append(float(shown))
    body = next(page.find_all("body")).text()
    stated = float(re.search(r"([\d.]+) percentage points", body).group(1))
    assert round(values[0] - values[1], 1) == stated, f"{values[0]} - {values[1]} is not the stated {stated}"
    headline = re.search(r"([\d.]+)\s*%", next(n for n in page.walk() if "big-number" in n.classes()).text()).group(1)
    assert float(headline) == values[0], "the large figure and the model bar must be the same number"


def test_links_to_the_evidence_notes_point_at_the_notes(page):
    notes = [d for d in page.find_all("details") if d.attrs.get("id") == "work-in-progress"]
    assert len(notes) == 1, "the evidence notes the page points to must exist"
    assert len(list(notes[0].find_all("li"))) >= 1
    evidence = next(s for s in page.find_all("section") if s.attrs.get("id") == "evidence")
    hrefs = [a.attrs.get("href") for a in evidence.find_all("a")]
    assert "#work-in-progress" in hrefs, "the headline result must keep its link to the evidence notes"


def test_links_into_this_repository_resolve(page):
    checked, problems = check_repository_links(outside_links(page), "siddhant61/StreamSense", REPO_ROOT, "master")
    assert checked >= 1
    assert not problems, "\n".join(problems)


def test_links_into_prosense_resolve(page):
    root = os.environ.get("PROSENSE_ROOT")
    if not root:
        pytest.skip("set PROSENSE_ROOT to a ProSense checkout to check the links into it")
    checked, problems = check_repository_links(outside_links(page), "siddhant61/ProSense", Path(root).resolve(), "main")
    assert checked >= 1
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "heading, slug",
    [
        ("🚀 Quick Start", "-quick-start"),
        ("Multimodal fusion", "multimodal-fusion"),
        ("Results (5-fold GroupKFold, 10 subjects)", "results-5-fold-groupkfold-10-subjects"),
        ("HRV stress → recovery (morning-breathing app)", "hrv-stress--recovery-morning-breathing-app"),
    ],
)
def test_github_slug(heading, slug):
    assert github_slug(heading) == slug
