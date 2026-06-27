# OpenCode Session — Person Architecture Inspection

Date: 2026-06-27
Model: GLM-5.2 via OpenRouter
Cost: about $0.13
Context: about 57k tokens
Branch: feature/person-architecture
Mode: read-only

## Outcome

OpenCode inspected the project and did not modify files.

## Key Findings

- Student is currently the central identity model.
- MealSubscription, Wallet, WalletTransaction, AttendanceRecord, AttendanceEvent, and FaceEmbedding are tightly coupled to Student.
- Meal pricing/confirmation logic still exists heavily in views.py and views_.py.
- Person architecture should be incremental.
- Do not delete Student early.

## Recommended Plan

Phase 0: Extract meal pricing/confirmation logic toward service layer.
Phase 1: Add Person and StudentProfile additively.
Phase 2: Add dual-write bridge.
Phase 3: Switch new code to Person.
Phase 4: Add StaffProfile.
Phase 5: Retire legacy Student coupling later.
