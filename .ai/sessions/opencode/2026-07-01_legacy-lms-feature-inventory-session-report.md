# Session Report — Legacy LMS Feature Inventory

Date: 2026-07-01
Mode: architecture analysis
Source: `edu-backend.tar.gz`

## Objective

Analyze the legacy BISK LMS backend as a business feature source and create a clean feature inventory to inform the future BISK_RFv4 foundation architecture.

## Findings

The backend archive contains substantial school ERP/LMS functionality, including students, guardians, families, teachers, academic years, terms, grades, classes, subjects, timetable entities, attendance/absence, finance/invoices/installments/receipts, admissions, marks/exams/interim/end-of-year reports, behaviour, chat, notifications, shop/items, clinic referrals, settings, automated tasks, and external/SAP integration hints.

## Decisions

- Treat the legacy LMS as a requirements/business-workflow source, not as code to copy.
- Do not require the large frontend archive yet.
- Use the backend feature inventory as input to `foundation_architecture.md`.
- Keep the implementation roadmap incremental and avoid rebuilding every legacy module immediately.

## Implementation

Created a Markdown inventory document:

- `legacy_lms_feature_inventory.md`

No BISK_RFv4 repository code was modified in this environment.

## Files Modified

Created in sandbox only:

- `/mnt/data/legacy_lms_feature_inventory.md`
- `/mnt/data/2026-07-01_legacy-lms-feature-inventory-session-report.md`

## Commands Executed

- Listed archive contents from `edu-backend.tar.gz`.
- Extracted selected backend source files excluding `.git`, fonts, images, and heavy/unnecessary artifacts.
- Reviewed model, route, service, migration, report, automated-task, and integration file names.
- Inspected representative model files such as `student-profile.js`, `user.js`, and `admissions.js`.

## Remaining Work

1. Use this inventory to improve the OpenCode prompt for `foundation_architecture.md`.
2. Create or update `docs/architecture/legacy_lms_feature_inventory.md` inside the BISK_RFv4 repository.
3. Create `docs/architecture/foundation_architecture.md` after reviewing the inventory.
4. Defer frontend upload unless dashboard/menu/UX analysis becomes necessary.

## Recommendations

Use the legacy backend to make the future foundation broader and more realistic, especially around AcademicYear, family/guardian relationships, finance, reporting, notifications, and admissions. Do not copy the old data model directly; design cleaner Django domain models around Person, roles, enrollments, assignments, and modular ERP domains.
