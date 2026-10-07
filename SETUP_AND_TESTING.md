# Project Setup and Local Testing Guide

This document provides all the necessary steps for a developer to set up the repository for the first time and to test the script locally.

## 1. One-Time Project Setup

To make the automation work, you must generate three secret keys and add them to your GitHub repository's settings.

### Step 1.1: Generate the Secrets

**1. Google Cloud credentials for Claude on Vertex AI**
The AI analysis uses LangChain's `ChatAnthropicVertex` integration. Authenticate
with Google Application Default Credentials locally (`gcloud auth application-default
login`) or configure the GitHub workflow with a service account/workload identity that
can invoke Claude on Vertex AI. Set `ANTHROPIC_VERTEX_PROJECT_ID` and
`CLOUD_ML_REGION` for the target project and region.

The log-analysis and final-synthesis models are configured separately in
[`ai_models.json`](ai_models.json). Set `AI_MODEL_CONFIG` to use a different config
file without editing the repository default.

**2. GitHub Personal Access Token**
This token lets the inspector automatically compare failure signatures with open
issues in `fedora-iot/iot-distro` and report potential prior reports in Slack.
It uses the AI's structured analysis plus raw-log evidence, and compares only the
meaningful bug-report fields rather than reproduction-template text. The matching
decision remains locally verified against the logs.

* **How to get it:** Follow the guide to create a "classic" Personal Access Token.
* **Guide:** [Creating a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)
  * **Scope/Permission needed:** You only need to grant the `public_repo` scope.
  * **Note:** Copy the token immediately after generation. You will not see it again.

**3. Slack Incoming Webhook URL**
This URL is used to post the final summary report to a Slack channel.
Note: Since a slack channel is already created, all the developers need not
create a new webhook. The same can be copied to your settings.
SLACK_WEBHOOK_URL=""

(still mentioning the step if needed to create from scratch)
* **How to get it:** Follow the guide to create a new Incoming Webhook.
* **Guide:** [Sending messages using Incoming Webhooks](https://api.slack.com/messaging/webhooks)
  1. Create a minimal Slack "App".
  2. Activate "Incoming Webhooks" for the app.
  3. Add a new webhook to your desired workspace and channel. (This may require approval from your Slack workspace admin).
  4. Copy the Webhook URL (it starts with `https://hooks.slack.com/...`).

### Step 1.2: Add Secrets to GitHub

1. In your GitHub repository, go to **`Settings`**.
2. In the left sidebar, navigate to **`Security` > `Secrets and variables` > `Actions`**.
3. Ensure you are on the **`Secrets`** tab.
4. Click **`New repository secret`** for each of the keys you generated. **The names must be an exact match**:
   * `MY_GITHUB_TOKEN`: The `ghp_...` token you got from GitHub.
   * `SLACK_WEBHOOK_URL`: The webhook URL you got from Slack.

---

## 2. Local Testing

### Step 2.1: Create a Virtual Environment

It is a best practice to use a virtual environment to manage project-specific dependencies.
Python 3.14+ may show an upstream LangChain/Pydantic compatibility warning during AI
initialization; the inspector leaves this warning visible.

```bash
# From your project's root directory
python3 -m venv .venv
```

### Step 2.2: Activate the Virtual Environment

```bash
source .venv/bin/activate
```
Your terminal prompt should now be prefixed with (.venv).

### Step 2.3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2.4: Create a Local .env File for Secrets
The script needs your secret keys to run locally. In the root directory of your project, create a new file named .env. 
Add your secrets to this file using the format VARIABLE_NAME="value".
This file is for local development only. Do not commit it.

ANTHROPIC_VERTEX_PROJECT_ID="your-google-cloud-project"
CLOUD_ML_REGION="global"
# Optional: path to a different JSON model configuration
# AI_MODEL_CONFIG="/path/to/ai_models.json"
MY_GITHUB_TOKEN="ghp_YourGitHubTokenGoesHere"
SLACK_WEBHOOK_URL="[https://hooks.slack.com/services/your/webhook/url/here](https://hooks.slack.com/services/your/webhook/url/here)"

### Step 2.5: Run the Script
You are now ready to run the script locally.

```bash
python check_fedora_iot.py
```
