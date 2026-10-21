# Changelog

All notable changes to Reimagine are recorded here. The format follows
Keep a Changelog; the version is the app version in `Cargo.toml`
`[workspace.package]` and in `src-tauri/tauri.conf.json`.

A breaking commit (`!` before the colon, or a `BREAKING CHANGE:` footer) must
add its note here in the same push - the commit-msg hook and the CI guards job
both check it.

## [Unreleased]

### Security

- Locked h2 0.4.20, quinn-proto 0.11.19 and rustls 0.23.46 (with
  rustls-webpki 0.103.16 and aws-lc-rs 1.18.2) to clear their RustSec
  advisories, and chacha20 0.10.2 in place of the yanked 0.10.1.
- Added a `cargo deny` policy (`deny.toml`) for advisories, licenses, bans
  and sources, run by the CI guards job.

## [0.1.0] - baseline

- Baseline entry: this file was introduced with the development-workflow
  alignment. Earlier history lives in `git log`; the app version is 0.1.0 and
  has not been released as a public bundle yet.
