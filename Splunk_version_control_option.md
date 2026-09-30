# Splunk Version Control & Reproducibility (TOS)

**Jira:** AUT-537 · **Splunk app:** `wp_DSE` · **Support request:** RITM0339876 (pending)

## Background

TOS alerts and dashboards are created and edited directly in the Splunk UI. There is no version control, so changes aren't reviewed, there's no history of what changed, and nothing can be easily rebuilt if it's deleted or needs to be moved between dev and prod.

A request has been raised with Splunk BAU Support (RITM0339876) to ask what options are officially supported. Until we hear back, we'll use Option 1.

## Option 1: Keep definitions in Git and update Splunk manually (current approach)

The TOS repo becomes the source of truth. Every alert and dashboard has a copy of its definition in the repo, and any change goes through a PR before it's made in Splunk.

**What to store:**

- **Dashboards:** the source from *Edit → Source* in Splunk (XML or JSON).
- **Alerts:** the SPL query, schedule, trigger condition, throttling and email recipients.

**Folder structure (TOS repo):**

```
splunk/
  alerts/
  dashboards/
  CHANGELOG.md
```

**Process for making a change:**

1. Update the file in the TOS repo and raise a PR
2. Get it reviewed and merged
3. Make the same change in the Splunk UI
4. Add an entry to `CHANGELOG.md` (date, who, what, Jira ticket)

**Pros:** works now with our current access, changes get reviewed, and we have history and a way to rebuild anything.
**Cons:** relies on people following the process; if someone edits directly in the UI, Git and Splunk can go out of sync.

## Other options (all need Splunk BAU approval)

These are more automated, but each depends on access or support from the Splunk BAU team, so they're included in the RITM request.

- **Automated backup:** a script pulls alert and dashboard definitions from Splunk through its API and saves them to Git on a schedule. Needs API access.
- **Deploy from Git:** changes merged in Git are pushed to Splunk automatically (script or Terraform). Needs API access with write permissions.
- **Splunk app:** alerts and dashboards are packaged as a Splunk app and installed by the Splunk admins. Every change would go through them.

## Keeping things reproducible

- Use a macro for the index name so the same search works in dev and prod.
- Store the full alert setup in Git (schedule, triggers, recipients), not just the query.
- Use consistent names, e.g. `TOS <Purpose> - <Env>`.

## Next steps

- Use Option 1 for now.
- Revisit once Splunk BAU Support responds to RITM0339876.
