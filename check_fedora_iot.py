import requests
from bs4 import BeautifulSoup
import sys
import re
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import time
import json
from email.utils import parsedate_to_datetime

from ai_analysis import (
    analysis_match_terms,
    format_issue_report,
    parse_ai_json,
    analyze_log_source,
    create_langchain_models,
    synthesize_log_analyses,
)
from utils import extract_signal_windows, prioritize_items

load_dotenv()

# --- Configurations ---
COMPOSE_BASE_URL = "https://kojipkgs.fedoraproject.org/compose/iot/"
OPENQA_API_URL = "https://openqa.fedoraproject.org/api/v1"
QUAY_API_URL = "https://quay.io/api/v1"
QUAY_REPOS = ["fedora/fedora-iot", "fedora/fedora-bootc"]
ISSUE_REPOSITORY = os.getenv("ISSUE_REPOSITORY", "fedora-iot/iot-distro")
ISSUE_LIST_URL = f"https://github.com/{ISSUE_REPOSITORY}/issues"
MAX_RELATED_ISSUES = 3
MAX_LOG_ANALYSES = 8
RETRY_COUNT = 3
RETRY_DELAY_SECONDS = 60
RUN_URL = f"https://github.com/{os.getenv('GITHUB_REPOSITORY', 'your/repo')}/actions/runs/{os.getenv('GITHUB_RUN_ID', 'local')}"

# --- AI Configuration (LangChain + Claude on Vertex AI) ---
VERTEX_PROJECT_ID = os.getenv("ANTHROPIC_VERTEX_PROJECT_ID", "itpc-ca-XXXXXXXXXX")
VERTEX_REGION = os.getenv("CLOUD_ML_REGION", "global")
AI_MODEL_CONFIG = os.getenv(
    "AI_MODEL_CONFIG", os.path.join(os.path.dirname(__file__), "ai_models.json")
)
ai_models = None

try:
    ai_models = create_langchain_models(AI_MODEL_CONFIG, VERTEX_PROJECT_ID, VERTEX_REGION)
    print(
        "AI configured through LangChain "
        f"(project={VERTEX_PROJECT_ID}, region={VERTEX_REGION}, config={AI_MODEL_CONFIG})"
    )
except Exception as e:
    print(f"Warning: Could not configure LangChain AI models: {e}. AI analysis will be disabled.")

# --- GitHub API ---
try:
    from github import Github, Auth
    MY_GITHUB_TOKEN = os.getenv("MY_GITHUB_TOKEN")
    g = Github(auth=Auth.Token(MY_GITHUB_TOKEN))
    _ = g.get_user().login
    print("GitHub client configured successfully.")
except Exception:
    from github import Github
    print("Warning: MY_GITHUB_TOKEN not found or invalid. Using unauthenticated GitHub API.")
    g = Github()

# --- Slack ---
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# ============================================================
# PART 1: DETERMINISTIC CHECKS
# ============================================================

def get_url_content(url, silent=False):
    """Fetch text content from a URL. Set silent=True to suppress 404 error messages."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        if not silent:
            print(f"    -> ERROR: Could not fetch {url}. Reason: {e}")
        return None


def get_all_compose_links():
    """Fetch the compose index page with retry logic."""
    for attempt in range(RETRY_COUNT):
        try:
            print(f"Fetching compose index (attempt {attempt + 1}/{RETRY_COUNT})...")
            response = requests.get(COMPOSE_BASE_URL, timeout=60)
            response.raise_for_status()
            print("Compose index fetched successfully.")
            return BeautifulSoup(response.text, 'html.parser').find_all('a')
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < RETRY_COUNT - 1:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
    print("CRITICAL: Could not fetch the compose index after multiple attempts.")
    return None


def get_fedora_release_info():
    """Query Bodhi API to determine stable and rawhide versions.

    Returns (stable_version, rawhide_version) as integers, or (None, None) on failure.
    """
    try:
        response = requests.get(
            "https://bodhi.fedoraproject.org/releases/",
            params={"exclude_archived": "true", "rows_per_page": "50"},
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()

        current_versions = []
        rawhide_version = None
        for r in response.json().get("releases", []):
            if r.get("id_prefix") != "FEDORA" or not r.get("version", "").isdigit():
                continue
            ver = int(r["version"])
            if r.get("branch") == "rawhide":
                rawhide_version = ver
            elif r.get("state") == "current":
                current_versions.append(ver)

        stable = max(current_versions) if current_versions else None
        print(f"Bodhi API: stable=F{stable}, rawhide=F{rawhide_version}")
        return stable, rawhide_version
    except Exception as e:
        print(f"Warning: Could not query Bodhi API: {e}")
        return None, None


def detect_active_versions(all_links):
    """Auto-detect which Fedora IoT versions to inspect.

    Uses the Bodhi API to determine stable, then keeps stable-1 through rawhide.
    Drops anything older than stable-1.
    """
    stable, _ = get_fedora_release_info()

    cutoff_str = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y%m%d')
    version_pattern = re.compile(r'Fedora-IoT-(\d+)-(\d{8})\.\d+/')
    composed_versions = set()

    for link in all_links:
        href = link.get('href', '')
        match = version_pattern.match(href)
        if match:
            version, date_str = match.groups()
            if date_str >= cutoff_str:
                composed_versions.add(int(version))

    print(f"Versions composed in last 7 days: {sorted(composed_versions, reverse=True)}")

    if not composed_versions:
        print("No versions composed in the last 7 days.")
        return []

    if stable:
        min_version = stable - 1
        active = [v for v in composed_versions if v >= min_version]
        dropped = [v for v in composed_versions if v < min_version]
        if dropped:
            print(f"Dropped EOL versions: {sorted(dropped, reverse=True)}")
    else:
        print("Warning: Could not determine stable version, keeping top 3")
        active = sorted(composed_versions, reverse=True)[:3]

    sorted_versions = [str(v) for v in sorted(active, reverse=True)]
    print(f"Active versions to inspect: {sorted_versions}")
    return sorted_versions


def find_compose_for_date(version, all_links, date_str):
    """Find today's compose directory for a given version. Returns (compose_url, build_name)."""
    pattern = re.compile(f"Fedora-IoT-{version}-{date_str}\\.\\d+\\/")
    version_links = [link.get('href') for link in all_links if pattern.match(link.get('href', ''))]
    if not version_links:
        return None, None
    latest_compose_dir = sorted(version_links)[-1]
    build_name = latest_compose_dir.rstrip('/')
    compose_url = f"{COMPOSE_BASE_URL}{latest_compose_dir}"
    print(f"  -> Found compose: {build_name}")
    return compose_url, build_name


def check_compose_status(compose_url):
    """Check the STATUS file for a compose. Returns the status string or None."""
    status_content = get_url_content(f"{compose_url}STATUS")
    if not status_content:
        return None
    return status_content.strip()


def get_compose_latest_artifact_time(compose_url):
    """Return the latest artifact timestamp recorded in compose images metadata.

    Pungi's images.json stores an mtime for each produced artifact.  The latest
    one is the best timestamp available for when this compose's image output
    finished; it lets us state whether a Quay tag changed afterwards.
    """
    content = get_url_content(f"{compose_url}compose/metadata/images.json", silent=True)
    if not content:
        return None

    try:
        metadata = json.loads(content)
    except json.JSONDecodeError:
        print("    -> Could not parse compose images.json metadata")
        return None

    mtimes = []

    def collect_mtimes(value):
        if isinstance(value, dict):
            mtime = value.get("mtime")
            if isinstance(mtime, (int, float)):
                mtimes.append(mtime)
            for child in value.values():
                collect_mtimes(child)
        elif isinstance(value, list):
            for child in value:
                collect_mtimes(child)

    collect_mtimes(metadata.get("payload", metadata))
    if not mtimes:
        print("    -> Compose images.json did not contain artifact timestamps")
        return None

    return datetime.fromtimestamp(max(mtimes), timezone.utc).isoformat()


def quay_updated_after_compose(compose_artifact_time, quay_last_modified):
    """Return whether a Quay tag changed after this compose's last artifact."""
    if not compose_artifact_time or not quay_last_modified or quay_last_modified == "unknown":
        return None

    try:
        compose_time = datetime.fromisoformat(compose_artifact_time).astimezone(timezone.utc)
        quay_time = parsedate_to_datetime(quay_last_modified)
        # Quay uses -0000, which Python represents as a naive datetime even
        # though the API timestamp is UTC.  Never let the runner's local zone
        # change the Yes/No comparison.
        if quay_time.tzinfo is None:
            quay_time = quay_time.replace(tzinfo=timezone.utc)
        quay_time = quay_time.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError, IndexError):
        return None

    return quay_time > compose_time


def check_openqa_results(version, build_name):
    """Query openQA API for IoT test results."""
    results = {
        "passed": 0, "failed": 0, "softfailed": 0,
        "running": 0, "scheduled": 0,
        "failed_tests": [], "softfailed_tests": [],
        "url": None, "total": 0
    }
    if not build_name:
        return results

    try:
        response = requests.get(
            f"{OPENQA_API_URL}/jobs",
            params={"distri": "fedora", "version": version, "build": build_name, "latest": "1", "scope": "current"},
            timeout=30,
        )
        response.raise_for_status()

        for job in response.json().get("jobs", []):
            state = job.get("state", "")
            result = job.get("result", "")
            test_name = job.get("test", "unknown")
            arch = job.get("settings", {}).get("ARCH", "unknown")

            if state == "done":
                if result == "failed":
                    results["failed"] += 1
                    results["failed_tests"].append(f"{test_name} ({arch})")
                elif result == "softfailed":
                    results["softfailed"] += 1
                    results["softfailed_tests"].append(f"{test_name} ({arch})")
                elif result == "passed":
                    results["passed"] += 1
            elif state in ("running", "scheduled"):
                results[state] += 1

        results["total"] = len(response.json().get("jobs", []))
        results["url"] = (
            f"https://openqa.fedoraproject.org/tests/overview?"
            f"distri=fedora&version={version}&build={build_name}&groupid=1&groupid=5"
        )
    except Exception as e:
        print(f"    -> openQA API error: {e}")

    return results


def check_quay_container(repo, tag):
    """Check if a container tag exists on Quay.io and when it was last updated."""
    try:
        response = requests.get(f"{QUAY_API_URL}/repository/{repo}/tag/?specificTag={tag}", timeout=30)
        response.raise_for_status()
        tags = response.json().get("tags", [])
        if tags:
            tag = tags[0]
            return {
                "exists": True,
                "manifest_digest": tag.get("manifest_digest"),
                "last_modified": tag.get("last_modified", "unknown"),
            }
        return {"exists": False}
    except Exception as e:
        print(f"    -> Quay.io API error for {repo}:{tag}: {e}")
        return {"exists": False, "error": str(e)}


# ============================================================
# PART 2: AUTOMATED ISSUE CORRELATION + AI FAILURE ANALYSIS
# ============================================================

ISSUE_MATCH_STOP_WORDS = {
    "about", "after", "also", "build", "cannot", "command", "compose",
    "container", "could", "error", "exception", "failed", "failure", "fedora",
    "file", "from", "image", "into", "iot", "missing", "package", "root",
    "service", "status", "that", "the", "this", "with",
}
ISSUE_SIGNAL = re.compile(
    r"(?:\berror\b|\bfailed\b|\bfailure\b|\bexception\b|traceback|"
    r"not found|missing|cannot|no such file|exit code)",
    re.IGNORECASE,
)
ISSUE_REPORT_SECTIONS = {
    "describe the bug", "expected behavior", "os version", "additional context",
}


def normalized_issue_terms(text):
    """Return comparable, non-generic tokens while preserving package/service names."""
    terms = re.findall(r"[a-z][a-z0-9_.+:-]{3,}", text.lower())
    return {
        term.strip("._+:-") for term in terms
        if term.strip("._+:-") not in ISSUE_MATCH_STOP_WORDS
    }


def issue_match_terms(logs):
    """Extract distinctive terms from failure lines for deterministic issue matching."""
    failure_lines = []
    for content in logs.values():
        for line in content.splitlines():
            if ISSUE_SIGNAL.search(line) and "/sys/fs/selinux/" not in line:
                failure_lines.append(line)

    # Use a bounded recent sample so normal log messages do not become a signature.
    failure_text = "\n".join(failure_lines[-100:]).lower()
    return {term for term in normalized_issue_terms(failure_text) if not term.isdigit()}


def issue_report_text(title, body):
    """Use only meaningful fields from the iot-distro bug-report template."""
    sections = []
    active_section = None
    for line in (body or "").splitlines():
        heading = line.strip().strip("#").strip().strip("*").strip().rstrip(":").lower()
        if heading in ISSUE_REPORT_SECTIONS:
            active_section = heading
            continue
        if heading in {"to reproduce", "screenshots"}:
            active_section = None
            continue
        if active_section and line.strip() and not line.lstrip().startswith("Please replace this line"):
            sections.append(line)

    # Older issues may not use the current template, so retain their body as a fallback.
    relevant_text = "\n".join(sections) if sections else (body or "")
    return f"{title or ''}\n{relevant_text}"


def find_related_open_issues(logs, analysis_result=None):
    """Match structured analysis and raw log evidence with open issues.

    The AI improves terminology, but a candidate must still share raw-log evidence
    with an existing issue before it is reported.
    """
    log_terms = issue_match_terms(logs)
    if not log_terms:
        return [], None
    ai_terms = analysis_match_terms(analysis_result, normalized_issue_terms)

    try:
        repository = g.get_repo(ISSUE_REPOSITORY)
        matches = []
        for issue in repository.get_issues(state="open", sort="updated", direction="desc"):
            issue_terms = normalized_issue_terms(issue_report_text(issue.title, issue.body))
            title_terms = normalized_issue_terms(issue.title or "")
            log_shared_terms = sorted(log_terms & issue_terms)
            ai_shared_terms = sorted(ai_terms & issue_terms)
            shared_terms = sorted(set(log_shared_terms) | set(ai_shared_terms))
            score = sum(min(len(term), 12) for term in shared_terms)
            distinctive_match = any(len(term) >= 12 for term in log_shared_terms)
            narrow_title_match = (
                len(title_terms) <= 2
                and any(len(term) >= 5 and term in title_terms for term in log_shared_terms)
            )
            raw_log_match = distinctive_match or narrow_title_match or (
                len(log_shared_terms) >= 2
                and sum(min(len(term), 12) for term in log_shared_terms) >= 12
            )
            analysis_assisted_match = bool(log_shared_terms) and len(shared_terms) >= 2 and score >= 12
            if not (raw_log_match or analysis_assisted_match):
                continue

            matches.append({
                "number": issue.number,
                "title": issue.title,
                "url": issue.html_url,
                "matched_log_terms": log_shared_terms[:4],
                "matched_analysis_terms": ai_shared_terms[:4],
                "confidence": "high" if raw_log_match else "medium",
                "score": score,
            })

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:MAX_RELATED_ISSUES], None
    except Exception as e:
        print(f"    -> Open issue lookup failed: {e}")
        return [], str(e)


def format_open_issue_check(related_issues, lookup_error):
    """Format the automatic open-issue check for Slack or CLI output."""
    if lookup_error:
        return f"*Open-issue check:* unavailable ({lookup_error[:160]})"
    if not related_issues:
        return (
            f"*Open-issue check:* no matching open issue found in "
            f"<{ISSUE_LIST_URL}|fedora-iot/iot-distro>."
        )

    formatted = []
    for issue in related_issues:
        log_terms = ", ".join(issue["matched_log_terms"])
        ai_terms = ", ".join(issue["matched_analysis_terms"])
        match_text = f"log: {log_terms}"
        if ai_terms:
            match_text += f"; analysis: {ai_terms}"
        formatted.append(
            f"<{issue['url']}|#{issue['number']} {issue['title']}> "
            f"({issue['confidence']} confidence; {match_text})"
        )
    return "*Potential related open issues:* " + "; ".join(formatted)

def find_koji_task_urls(compose_url):
    """Find Koji task URLs from osbuild/ or koji-tasks/ log directories."""
    # Try osbuild/ directory (watch-task logs with embedded Koji URLs)
    print(f"    -> Checking osbuild log directory...")
    dir_content = get_url_content(f"{compose_url}logs/global/osbuild/", silent=True)
    if dir_content:
        soup = BeautifulSoup(dir_content, 'html.parser')
        koji_urls = []
        for log_link in soup.find_all('a', href=re.compile(r'IoT-\d+-watch-task\.log$')):
            log_content = get_url_content(f"{compose_url}logs/global/osbuild/{log_link['href']}")
            if log_content:
                match = re.search(r'(https://koji\.fedoraproject\.org/koji/taskinfo\?taskID=\d+)', log_content)
                if match:
                    koji_urls.append(match.group(1))
        if koji_urls:
            print(f"      -> Found {len(koji_urls)} Koji task URL(s) from osbuild logs")
            return koji_urls

    # Try koji-tasks/ directory (task ID files)
    print(f"    -> Checking koji-tasks directory...")
    dir_content = get_url_content(f"{compose_url}logs/global/koji-tasks/", silent=True)
    if dir_content:
        soup = BeautifulSoup(dir_content, 'html.parser')
        koji_urls = [
            f"https://koji.fedoraproject.org/koji/taskinfo?taskID={link['href']}"
            for link in soup.find_all('a', href=re.compile(r'^\d+$'))
        ]
        if koji_urls:
            print(f"      -> Found {len(koji_urls)} Koji task(s): {[u.split('=')[-1] for u in koji_urls]}")
            return koji_urls

    print("    -> No Koji task URLs found.")
    return []


def get_koji_task_logs(koji_task_url):
    """Extract all available log data from a Koji task page.

    Returns combined log content from compose-status.json, build.log, root.log,
    do_mounts.log, and runroot.log.
    """
    print(f"    -> Drilling into Koji Task: {koji_task_url}")
    page_content = get_url_content(koji_task_url)
    if not page_content:
        return None

    soup = BeautifulSoup(page_content, 'html.parser')

    # compose-status.json is self-contained for osbuild failures — return it directly
    json_link = soup.find('a', href=re.compile(r'.*compose-status\.json$'))
    if json_link:
        json_url = json_link['href']
        if not json_url.startswith('http'):
            json_url = "https://kojipkgs.fedoraproject.org/" + json_url
        print(f"      -> Found compose-status.json")
        content = get_url_content(json_url)
        if content:
            try:
                return json.dumps(json.loads(content), indent=2)
            except json.JSONDecodeError:
                pass

    # Collect all relevant log files (tail of each — errors are at the end)
    collected = []
    for log_name in ["build.log", "root.log", "do_mounts.log", "runroot.log"]:
        log_link = soup.find('a', string=re.compile(f'^{re.escape(log_name)}$'))
        if log_link:
            log_url = log_link['href']
            if not log_url.startswith('http'):
                log_url = "https://kojipkgs.fedoraproject.org/" + log_url
            print(f"      -> Fetching {log_name}...")
            content = get_url_content(log_url)
            if content and len(content.strip()) > 10:
                collected.append(f"=== {log_name} ===\n{content[-4000:]}")

    if collected:
        return "\n\n".join(collected)

    return None


def collect_failure_logs(compose_url):
    """Gather all available log data from a failed compose. Returns a dict of log sources."""
    logs = {}

    # pungi.global.log (high-level overview)
    print("  --- Collecting pungi.global.log ---")
    content = get_url_content(f"{compose_url}logs/global/pungi.global.log")
    if content:
        logs["pungi.global.log"] = content[-5000:]

    # deliverables.json (what exactly failed)
    print("  --- Collecting deliverables.json ---")
    content = get_url_content(f"{compose_url}logs/global/deliverables.json", silent=True)
    if content:
        logs["deliverables.json"] = content

    # Per-architecture runroot logs (ostree-container failures)
    print("  --- Collecting per-architecture runroot logs ---")
    for arch in ["x86_64", "aarch64", "s390x", "ppc64le"]:
        content = get_url_content(
            f"{compose_url}logs/{arch}/IoT/ostree-container-1/runroot.log", silent=True
        )
        if content and len(content.strip()) > 50:
            print(f"      -> Found runroot.log for {arch} ({len(content)} bytes)")
            logs[f"runroot.log ({arch})"] = content

    # Koji task logs (build.log, root.log, do_mounts.log, compose-status.json)
    print("  --- Collecting Koji task logs ---")
    for koji_task_url in find_koji_task_urls(compose_url)[:3]:
        task_id = koji_task_url.split('=')[-1]
        task_logs = get_koji_task_logs(koji_task_url)
        if task_logs:
            logs[f"koji_task_{task_id}"] = task_logs
            break  # One good task is enough

    return logs


def affected_arches_from_logs(logs):
    """Read affected architectures from deliverables data when it is available."""
    deliverables = logs.get("deliverables.json", "")
    known_arches = ["x86_64", "aarch64", "s390x", "ppc64le"]
    return [arch for arch in known_arches if re.search(rf"\b{re.escape(arch)}\b", deliverables)]


def log_analysis_order(logs):
    """Choose a deterministic order from metadata to root-cause logs."""
    affected_arches = affected_arches_from_logs(logs)
    matching_runroots = [
        f"runroot.log ({arch})" for arch in affected_arches
        if f"runroot.log ({arch})" in logs
    ]
    remaining_runroots = [
        source for source in logs if source.startswith("runroot.log")
        and source not in matching_runroots
    ]
    koji_sources = [source for source in logs if source.startswith("koji_task_")]

    ordered = [
        source for source in ("deliverables.json", "pungi.global.log") if source in logs
    ]
    ordered.extend(matching_runroots)
    ordered.extend(koji_sources)
    ordered.extend(remaining_runroots)
    return ordered[:MAX_LOG_ANALYSES]


def run_staged_log_analysis(logs):
    """Analyze each relevant source in bounded rounds before final synthesis.

    Metadata is analyzed first, then the affected architecture's runroot logs,
    followed by the implicated Koji task and any remaining architectures. A per-log
    analysis may request an available source; that request only reprioritizes the
    remaining deterministic plan and cannot fetch arbitrary data.
    """
    pending_sources = log_analysis_order(logs)
    analyses = []

    while pending_sources and len(analyses) < MAX_LOG_ANALYSES:
        source = pending_sources.pop(0)
        excerpt = extract_signal_windows(
            logs[source], ISSUE_SIGNAL, context_lines=2, max_windows=4, max_chars=8000
        )
        print(f"  --- Analyzing {source} ({len(excerpt)} chars of relevant context) ---")
        result = analyze_log_source(ai_models["log_analysis"], source, excerpt, pending_sources)
        analyses.append(result)
        pending_sources = prioritize_items(
            pending_sources, result.get("requested_next_sources", [])
        )

    return analyses


def log_source_url(compose_url, source):
    """Resolve a collected source name to a browser-visible log or Koji task URL."""
    if source == "pungi.global.log":
        return f"{compose_url}logs/global/pungi.global.log"
    if source == "deliverables.json":
        return f"{compose_url}logs/global/deliverables.json"
    arch_match = re.fullmatch(r"runroot\.log \(([^)]+)\)", source)
    if arch_match:
        arch = arch_match.group(1)
        return f"{compose_url}logs/{arch}/IoT/ostree-container-1/runroot.log"
    task_match = re.fullmatch(r"koji_task_(\d+)", source)
    if task_match:
        return f"https://koji.fedoraproject.org/koji/taskinfo?taskID={task_match.group(1)}"
    return f"{compose_url}logs/"


def verified_log_snapshot(logs, evidence):
    """Find a small real-log excerpt nearest to the AI's claimed evidence."""
    claimed_source = evidence.get("source", "") if isinstance(evidence, dict) else ""
    claimed_excerpt = evidence.get("excerpt", "") if isinstance(evidence, dict) else ""
    search_sources = [(claimed_source, logs[claimed_source])] if claimed_source in logs else list(logs.items())
    claimed_terms = normalized_issue_terms(claimed_excerpt)
    best_match = None

    for source, content in search_sources:
        lines = content.splitlines()
        for index, line in enumerate(lines):
            line_terms = normalized_issue_terms(line)
            matched_terms = claimed_terms & line_terms
            if claimed_terms and not matched_terms:
                continue
            score = len(matched_terms)
            if ISSUE_SIGNAL.search(line):
                score += 1
            if score and (best_match is None or score > best_match[0]):
                best_match = (score, source, lines, index)

    if not best_match:
        return None, None

    _, source, lines, index = best_match
    snapshot = "\n".join(lines[max(0, index - 1):index + 2]).strip()
    return source, snapshot[:600]


def format_verified_evidence(compose_url, logs, evidence):
    """Render only an evidence snapshot that can be traced back to collected logs."""
    source, snapshot = verified_log_snapshot(logs, evidence)
    if not snapshot:
        return "*Verified log evidence:* no matching excerpt could be located in collected logs."

    safe_snapshot = snapshot.replace("```", "'''")
    source_url = log_source_url(compose_url, source)
    return f"*Verified log evidence (<{source_url}|{source}>):*\n```{safe_snapshot}```"


def diagnose_failure(compose_url, version_name):
    """Collect logs, check existing issues, then run the AI diagnosis."""
    print(f"  Starting AI diagnosis for {version_name}...")

    logs = collect_failure_logs(compose_url)
    if not logs:
        return "_No log files found for analysis._"

    if not ai_models:
        print(f"  --- Checking open issues in {ISSUE_REPOSITORY} (raw-log fallback) ---")
        related_issues, issue_lookup_error = find_related_open_issues(logs)
        issue_check = format_open_issue_check(related_issues, issue_lookup_error)
        return f"_AI diagnosis unavailable._\n{issue_check}"

    print(f"  -> Collected {len(logs)} log sources")
    log_analyses = run_staged_log_analysis(logs)
    print(f"  --- Synthesizing {len(log_analyses)} per-log analyses ---")
    raw_analysis = synthesize_log_analyses(
        ai_models["synthesis"], log_analyses, [analysis["source"] for analysis in log_analyses]
    )
    print(f"  Final AI result: {raw_analysis}")

    try:
        result = parse_ai_json(raw_analysis)
    except (json.JSONDecodeError, ValueError):
        print(f"  --- Checking open issues in {ISSUE_REPOSITORY} (raw-log fallback) ---")
        related_issues, issue_lookup_error = find_related_open_issues(logs)
        issue_check = format_open_issue_check(related_issues, issue_lookup_error)
        return f"*AI Diagnosis:*\n```{raw_analysis[:2500]}```\n{issue_check}"

    print(f"  --- Checking open issues in {ISSUE_REPOSITORY} ---")
    related_issues, issue_lookup_error = find_related_open_issues(logs, result)
    issue_check = format_open_issue_check(related_issues, issue_lookup_error)

    # Format output - simplified for Slack readability
    severity = result.get('severity', 'unknown')
    evidence = result.get("evidence", [])
    remediation = result.get("remediation", "")
    investigation_reason = result.get("investigation_reason", "")

    diagnosis_parts = [f"*Severity:* {severity}", format_issue_report(result, compose_url, version_name)]

    if evidence:
        first_evidence = evidence[0] if isinstance(evidence[0], dict) else {}
        diagnosis_parts.append(format_verified_evidence(compose_url, logs, first_evidence))
    else:
        diagnosis_parts.append("*Verified log evidence:* no evidence excerpt was returned by the AI.")

    if remediation:
        diagnosis_parts.append(f"*Remediation:* {remediation}")
    elif result.get("needs_human_investigation") and investigation_reason:
        diagnosis_parts.append(f"*Missing evidence:* {investigation_reason}")

    diagnosis_parts.append(issue_check)
    return "\n".join(diagnosis_parts)


# ============================================================
# OUTPUT: SLACK NOTIFICATION
# ============================================================

def send_slack_notification(blocks):
    """Send a structured Slack message, or print to stdout if no webhook configured."""
    for block in blocks:
        if block.get("type") == "section":
            text = block.get("text", {}).get("text", "")
            if len(text) > 2900:
                block["text"]["text"] = text[:2900] + "\n... _(truncated)_"

    # Disable Slack for testing - set DISABLE_SLACK=true to skip sending
    if os.getenv("DISABLE_SLACK", "false").lower() == "true":
        print("\n" + "=" * 60)
        print("SLACK DISABLED (DISABLE_SLACK=true)")
        print("=" * 60)
        return

    if not SLACK_WEBHOOK_URL:
        print("\n" + "=" * 60)
        print("SLACK PREVIEW (SLACK_WEBHOOK_URL not set)")
        print("=" * 60)
        for block in blocks:
            btype = block.get("type")
            if btype == "header":
                print(f"\n  {block['text']['text']}")
                print("  " + "-" * 40)
            elif btype == "section":
                print(f"\n{block['text']['text']}")
            elif btype == "context":
                for el in block.get("elements", []):
                    print(f"  {el.get('text', '')}")
            elif btype == "divider":
                print("-" * 40)
        print("=" * 60)
        return

    print("Sending summary to Slack...")
    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps({"blocks": blocks}),
            headers={'Content-Type': 'application/json'},
            timeout=30,
        )
        response.raise_for_status()
        print("Slack notification sent successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending Slack notification: {e}")


def format_slack_blocks(date_str, version_reports):
    """Build structured Slack blocks from version reports."""
    # Build simple text message for better readability
    status_emojis = {
        "FINISHED": ":white_check_mark:",
        "FINISHED_INCOMPLETE": ":warning:",
        "DOOMED": ":fire:",
        "STARTED": ":hourglass_flowing_sand:",
        "MISSING": ":x:",
    }

    message_lines = [
        f":newspaper: *Fedora IoT Compose Status - {date_str}*",
        ""
    ]

    for report in version_reports:
        emoji = status_emojis.get(report["status"], ":question:")
        version = report['version']
        status = report['status']

        # Build status line
        if status in ("FINISHED", "STARTED"):
            line = f"{emoji} *Fedora-IoT-{version}:* Compose {status.lower()}"
        elif status == "MISSING":
            line = f"{emoji} *Fedora-IoT-{version}:* No compose found for {date_str}"
        else:
            line = f"{emoji} *Fedora-IoT-{version}:* Failed with status {status}"

        message_lines.append(line)

        # Add compose URL if available
        if report.get("compose_url"):
            message_lines.append(f"   • <{report['compose_url']}|View compose directory>")

        # Add openQA summary
        oqa = report.get("openqa")
        if oqa and oqa.get("total", 0) > 0:
            oqa_parts = []
            if oqa["passed"] > 0:
                oqa_parts.append(f"{oqa['passed']} passed")
            if oqa["failed"] > 0:
                oqa_parts.append(f"*{oqa['failed']} failed*")
            if oqa["softfailed"] > 0:
                oqa_parts.append(f"{oqa['softfailed']} softfailed")
            if oqa["running"] > 0:
                oqa_parts.append(f"{oqa['running']} running")

            if oqa_parts:
                oqa_text = f"   • openQA: {', '.join(oqa_parts)}"
                if oqa.get("url"):
                    oqa_text += f" — <{oqa['url']}|View tests>"
                message_lines.append(oqa_text)

            # Show failed tests
            if oqa["failed_tests"]:
                failed_list = ", ".join(oqa["failed_tests"][:3])
                if len(oqa["failed_tests"]) > 3:
                    failed_list += f" +{len(oqa['failed_tests']) - 3} more"
                message_lines.append(f"   • Failed tests: {failed_list}")

        # Add immutable Quay digest when available; otherwise retain the timestamp
        # as the best available indication of when the mutable version tag changed.
        for repo, container in report.get("containers", {}).items():
            if not container.get("exists"):
                message_lines.append(f"   • Quay: `{repo}:{version}` tag not found")
                continue

            digest = container.get("manifest_digest")
            last_modified = container.get("last_modified", "unknown")
            if digest:
                message_lines.append(
                    f"   • Quay: `{repo}:{version}` → `{digest}` "
                    f"(last modified: {last_modified})"
                )
            else:
                message_lines.append(
                    f"   • Quay: `{repo}:{version}` last modified: {last_modified} "
                    "(manifest digest unavailable)"
                )

            updated_after_compose = container.get("updated_after_compose")
            if updated_after_compose is True:
                answer = "*Yes*"
            elif updated_after_compose is False:
                answer = "*No*"
            else:
                answer = "Unknown (compose or Quay timestamp unavailable)"
            message_lines.append(f"   • Quay updated after compose run: {answer}")

        # Add AI diagnosis for failures
        if report.get("diagnosis"):
            message_lines.append("")
            message_lines.append(f"   :robot_face: *AI Analysis:*")
            message_lines.append(f"   {report['diagnosis']}")

        message_lines.append("")  # Blank line between versions

    # Add footer
    message_lines.append(f"<{RUN_URL}|View full GitHub Actions run log>")

    # Return as simple blocks
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(message_lines)}
        }
    ]


# ============================================================
# MAIN
# ============================================================

def inspect_version(version, all_links, current_date_str):
    """Run all checks for a single version. Returns a report dict."""
    print(f"\n{'='*40}")
    print(f"Inspecting Fedora IoT {version}...")

    report = {"version": version, "status": "MISSING", "compose_url": None}

    compose_url, build_name = find_compose_for_date(version, all_links, current_date_str)
    if not compose_url:
        print(f"  -> No compose found for {current_date_str}")
        return report

    report["compose_url"] = compose_url

    status = check_compose_status(compose_url)
    if not status:
        report["status"] = "UNKNOWN"
        return report
    report["status"] = status
    print(f"  -> Status: {status}")

    print("  -> Reading compose artifact timestamps...")
    report["compose_latest_artifact_time"] = get_compose_latest_artifact_time(compose_url)
    if report["compose_latest_artifact_time"]:
        print(f"     Latest artifact: {report['compose_latest_artifact_time']}")

    print(f"  -> Checking openQA results...")
    report["openqa"] = check_openqa_results(version, build_name)
    oqa = report["openqa"]
    print(f"     {oqa['passed']} passed, {oqa['failed']} failed, {oqa['softfailed']} softfailed")

    print(f"  -> Checking Quay.io containers...")
    report["containers"] = {repo: check_quay_container(repo, version) for repo in QUAY_REPOS}
    for container in report["containers"].values():
        container["updated_after_compose"] = quay_updated_after_compose(
            report["compose_latest_artifact_time"], container.get("last_modified")
        )

    if status in ("DOOMED", "FINISHED_INCOMPLETE"):
        report["diagnosis"] = diagnose_failure(compose_url, f"Fedora-IoT-{version}")

    return report


def save_daily_report(date_str, reports):
    """Save the daily report as a JSON artifact for meeting prep aggregation."""
    artifact_path = os.getenv("GITHUB_WORKSPACE", ".")
    report_file = os.path.join(artifact_path, f"daily-report-{date_str}.json")
    with open(report_file, "w") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "versions": reports,
        }, f, indent=2, default=str)
    print(f"Daily report saved to {report_file}")


def run_diagnose(build_name):
    """Run AI diagnosis on a specific compose build (e.g. Fedora-IoT-42-20260427.0)."""
    if not ai_models:
        print("ERROR: LangChain AI models not configured. Run 'gcloud auth application-default login' first.")
        sys.exit(1)

    compose_url = f"{COMPOSE_BASE_URL}{build_name}/"
    status = check_compose_status(compose_url)
    if not status:
        print(f"ERROR: Could not find compose at {compose_url}")
        sys.exit(1)

    print(f"Compose: {build_name} (status: {status})")
    result = diagnose_failure(compose_url, build_name)
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)


def main():
    # Handle --diagnose mode
    if len(sys.argv) >= 3 and sys.argv[1] == "--diagnose":
        run_diagnose(sys.argv[2])
        return

    print("Starting Fedora IoT Compose Inspection")
    print("=" * 50)

    # Allow date override for testing
    override_date = os.getenv("OVERRIDE_DATE")
    if override_date:
        current_date = datetime.strptime(override_date, '%Y%m%d').replace(tzinfo=timezone.utc)
        print(f"[TEST MODE] Using override date: {override_date}")
    else:
        current_date = datetime.now(timezone.utc)

    current_date_str = current_date.strftime('%Y%m%d')
    current_date_display = current_date.strftime('%Y-%m-%d')

    all_links = get_all_compose_links()
    if not all_links:
        send_slack_notification([{
            "type": "section",
            "text": {"type": "mrkdwn", "text": ":x: *CRITICAL:* Could not fetch the Fedora IoT compose index."}
        }])
        sys.exit(1)

    versions = detect_active_versions(all_links)
    if not versions:
        print("No active versions detected. Exiting.")
        sys.exit(1)

    reports = []
    has_failure = False
    for version in versions:
        report = inspect_version(version, all_links, current_date_str)
        reports.append(report)
        if report["status"] in ("DOOMED", "FINISHED_INCOMPLETE", "MISSING"):
            has_failure = True

    print(f"\n{'='*20} Summary {'='*20}")
    for r in reports:
        icon = {"FINISHED": "OK", "STARTED": "IN PROGRESS"}.get(r["status"], "ISSUE")
        print(f"  [{icon}] Fedora IoT {r['version']}: {r['status']}")

    send_slack_notification(format_slack_blocks(current_date_display, reports))
    save_daily_report(current_date_str, reports)

    if has_failure:
        print("\nInspection finished with one or more issues.")
    else:
        print("\nInspection finished successfully for all versions.")


if __name__ == "__main__":
    main()
