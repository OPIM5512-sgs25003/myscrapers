
# craigslist-scraper

A serverless web scraping pipeline built using Google Cloud Platform (GCP) and GitHub Actions. This project automatically scrapes data, deploys via CI/CD, and runs on a schedule.

---

## Project Overview

This project demonstrates a complete cloud-based data pipeline:

* Scrapes data using Python
* Deploys automatically using GitHub Actions
* Runs on Google Cloud Functions (Gen2 / Cloud Run)
* Executes on a schedule via Cloud Scheduler
* Stores results in Google Cloud Storage

---

## Architecture

GitHub → Cloud Functions (Gen2) → Cloud Run → Cloud Scheduler → Cloud Storage

1. Code is pushed to GitHub
2. GitHub Actions deploys the scraper
3. Cloud Function is exposed via HTTP
4. Cloud Scheduler triggers it every hour
5. Data is stored in Cloud Storage

---

## Technologies Used

* Python 3.12
* Google Cloud Functions (Gen2)
* Cloud Run
* Cloud Scheduler
* Cloud Storage
* GitHub Actions (CI/CD)
* IAM & Service Accounts

---

## Scheduling

The scraper runs automatically every hour using:

0 * * * *

Timezone: America/New_York

---

## Output

Data is saved in a Cloud Storage bucket. Each run creates a timestamped folder:

scrapes/
├── 20260404184624/
├── 20260404184632/
├── ...

---

## Deployment (Automated)

Deployment is handled via GitHub Actions:

* Authenticates with GCP (Workload Identity Federation)
* Deploys Cloud Function
* Configures IAM permissions
* Creates/updates Cloud Scheduler job

---

## Challenges & Fixes

During development, several issues were encountered:

* IAM permission errors (403)
* Missing roles:

  * iam.serviceAccountUser
  * iam.serviceAccountTokenCreator
* Cloud Scheduler permission issues
* HTTP 500 runtime errors

All issues were resolved through proper role assignments and API enablement.

---

## Final Status

* Deployment successful
* Scheduler working
* Data stored in Cloud Storage
* Logs confirm execution

---

## Course Context

This project was developed as part of:

OPIM 5512 – Data Science Using Python

---

## Author

Ates Boyaci
