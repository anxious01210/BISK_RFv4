# Legacy Lunch / Wallet / Attendance Investigation — BISK_RFv4

Date: 2026-07-03
Branch: feature/person-architecture
Status: Read-only investigation report. No code, models, or migrations changed.

---

## 1. What lunch/meal features currently exist?

The lunch domain is fully embedded inside `apps/attendance` and is operational:

| Feature | Location | Notes |
|---|---|---|
| `MealSubscription` (date-range entitlement) | `models.py:485` | FK to legacy `Student`; statuses active/cancelled/expired; priority + overlap validation in `clean()`. |
| `MealProfile` (product/pricing/policy) | `models.py:786` | Modes: `date_range` / `wallet`; insufficient-funds policy (`deny`/`allow_unpaid`/`allow_negative`); supervisor-override flags. |
| `MealProfilePeriod` (per-period price) | `models.py:851` | Links `MealProfile` ↔ `PeriodTemplate` with `price_iqd`. |
| `MealRecord` (per-attendance lunch service) | `models.py:874` | 1:1 with `AttendanceRecord`; statuses pending/confirmed/denied/unpaid/refunded/voided; snapshots price/discount/charge/balance; wallet-transaction FKs. |
| `MealSubscription.is_active_on(day)` | `models.py:575` | Convenience date check. |
| `Student.has_meal` flag | `models.py:60` | Transitional mirror, recalculated by `recalc_meal_flags_for_students`. |
| Meal-eligibility recalculation | `utils/meal.py` | `recalc_meal_flags_for_students`/`_all` set `Student.has_meal` from active subscriptions. |
| Meal-eligibility signal | `signals.py:29-45` | `post_save`/`post_delete` on `MealSubscription` triggers `has_meal` recalc via `transaction.on_commit`. |
| Postpaid profile auto-create | `views.py:39-55` | `_get_or_create_postpaid_profile` creates a reusable "Postpaid Wallet" `MealProfile`. |

## 2. What wallet/payment/balance features currently exist?

| Feature | Location | Notes |
|---|---|---|
| `Wallet` | `models.py:634` | O2O to legacy `Student`; `balance_iqd` integer; `is_active`. |
| `WalletTransaction` | `models.py:653` | Immutable ledger row; types topup/debit/refund/adjustment/unpaid; signed `amount_iqd`; `balance_before`/`after` snapshots; `reversed_transaction` self-FK; FK to `AttendanceRecord`. |
| Wallet admin | `admin.py:721-767` | `WalletAdmin` + `WalletTransactionAdmin` with search/autocomplete. |
| Wallet auto-create from dashboard | `views.py:1305-1312` | `enable_postpaid` view creates a wallet for the student if missing. |

**Critical:** wallet balance is **directly mutated in views** (see §8).

## 3. What supervisor dashboard features currently exist?

The meal supervisor dashboard is a working HTMX-based UI in `apps/attendance/views.py`:

| View | URL name | Purpose |
|---|---|---|
| `meal_page` | `meal_page` | Main dashboard page (`/dash/meal/`). |
| `meal_stream_rows` | `meal_stream_rows` | HTMX streaming rows of today's attendance records with resolved meal/wallet status. Modes: latest / all (paginated) / lastN. Filters: period, camera, eligible_only, min_score, q (search by h_code/name), date range, meal_bucket. |
| `meal_period_cards` | `meal_period_cards` | Period cards showing recognized/DR/wallet/blocked counts. |
| `meal_camera_health` | `meal_camera_health` | Camera freshness indicator (fresh/stale/no-data). |
| `confirm_record` | `confirm_record` | POST: confirm a meal — resolves subscription/profile/wallet, debits wallet or denies/unpaids based on policy. |
| `reverse_record` | `reverse_record` | POST: refund (wallet mode) or unconfirm/void (date-range/unpaid mode). |
| `enable_postpaid` | (commented out in urls) | POST: auto-create postpaid wallet + subscription. |

**Auth:** `meal_supervisor` Django Group check (`_is_meal_supervisor` at `views.py:558`).

**Key helper:** `_resolve_effective_meal_setup(student, meal_day, period_template)` at `views.py:100` resolves the active subscription + profile + wallet + block/credit-limit status for a student on a day.

## 4. What AI attendance features currently interact with lunch/meal logic?

| Integration point | Location | How it connects to lunch |
|---|---|---|
| `ingest_match(h_code, score, camera, ts, crop_path)` | `services.py:429` | API ingest entry; looks up `Student` by `h_code`, writes `AttendanceEvent` + `AttendanceRecord`. |
| `IngestView` | `api.py:310` | REST endpoint `/api/attendance/ingest/` accepting `h_code` + score + camera. |
| `AttendanceEvent` | `models.py:202` | FK to `Student`; produces `AttendanceRecord` which the meal dashboard reads. |
| `AttendanceRecord` | `models.py:172` | FK to `Student`; 1:1 `MealRecord` attached when lunch is confirmed. |
| `record_recognition(student, score, camera, ts, crop_path)` | `services.py:374` | Internal entry when a `Student` instance is already resolved. |
| `_write_from_match` | `services.py:79` | Core upsert: creates `AttendanceEvent` + upserts `AttendanceRecord` with pass-count logic. |
| Face embeddings | `models.py:347` (`FaceEmbedding`) | FK to `Student`; `uniq_active_embedding_per_student` constraint. |
| Recognition settings | `models.py:218` (`RecognitionSettings`) | Thresholds, dedup windows, pass-gap, max-periods-per-day. |

The meal dashboard reads `AttendanceRecord` rows produced by recognition and joins them to `MealRecord` to show the live lunch roster.

## 5. Which models are currently tied to the old Student model?

Every domain model in `apps/attendance` FKs to the legacy `attendance.Student`:

| Model | FK field | Related name |
|---|---|---|
| `AttendanceRecord` | `student` | (default) |
| `AttendanceEvent` | `student` | (default) |
| `FaceEmbedding` | `student` | `embeddings` |
| `MealSubscription` | `student` | `meal_subscriptions` |
| `Wallet` | `student` (O2O) | `wallet` |
| `WalletTransaction` | `student` | `wallet_transactions` |
| `MealRecord` | (indirect via `AttendanceRecord.student`) | — |

## 6. Which code depends on h_code or old student fields?

`h_code` is referenced in **133 lines across 14 files** (excluding migrations/`__pycache__`):

| File | Usage |
|---|---|
| `models.py` | `Student.h_code` field; `__str__`; `gallery_photo_relurl`; `full_name` fallback. |
| `services.py` | `ingest_match(h_code=...)`; `Student.objects.get(h_code=h_code)`. |
| `api.py` | `IngestView`/`EnrollView` accept `h_code`; filters `h_code_exact`/`h_code`/`students`; `search_fields`. |
| `admin.py` | `StudentAdmin` list_display/search/actions; `enroll_from_folder_action(st.h_code)`; gallery path resolution. |
| `resources.py` | `AttendanceRecordResource` exports `student__h_code`; `StudentResource` import-id = `h_code`. |
| `serializers.py` | `AttendanceRecordSerializer.h_code = student.h_code`. |
| `views.py` / `views_.py` | Row HTML renders `student.h_code`; search filter `student__h_code__icontains`. |
| `utils/embeddings.py` | Enrollment by `h_code`; gallery folder path. |
| `utils/media_paths.py` | `student_gallery_dir(h_code)`; face-gallery path layout. |
| Management commands | `enroll_from_folder`, `build_embeddings_pkl`, `sort_gallery_intake`, `assess_gallery_quality` all key on `h_code`. |

**Filesystem dependency:** `MEDIA_ROOT/face_gallery/<h_code>/` stores reference photos keyed by `h_code`.

## 7. What parts look stable and should be preserved?

| Stable part | Why preserve |
|---|---|
| `MealSubscription` model + `clean()` overlap logic | Working entitlement model; migrate FK target, not redesign. |
| `MealProfile` / `MealProfilePeriod` pricing/policy | Working product definition; generalize later, don't break now. |
| `MealRecord` status state machine + audit fields | Working lunch-attendance lifecycle; preserve as-is. |
| `WalletTransaction` immutable-ledger design | Good architecture; migrate FK target only. |
| `RecognitionSettings` solo config + `FaceEmbedding` constraint | Working recognition infra; don't touch. |
| `_resolve_effective_meal_setup` resolver | Core eligibility/balance/block logic; wrap as adapter, don't rewrite. |
| HTMX dashboard views + templates | Working supervisor UX; preserve behavior. |
| `recalc_meal_flags_for_students` + signal | Working `has_meal` sync; keep until enrollment-based eligibility replaces it. |
| Camera/streaming/scheduler infrastructure | Person-agnostic; only FK target changes. |

## 8. What parts look risky or inconsistent?

### 8.1 CRITICAL: Migration/DB schema drift

| Issue | Detail |
|---|---|
| **Both `0001_initial` and `0001_squashed_0021...` are recorded as applied** | The `django_migrations` table has both rows. The squash was **fake-applied** — it was recorded without actually running. |
| **`confirmed` field (squash 0020) never applied to DB** | The squash includes `AddField('attendancerecord','confirmed')` but the DB column does not exist. The current `models.py` also omits it, so `makemigrations --check` says "no changes" (models match migration state, not DB). |
| **`lunch_*` fields (migration 0026) never applied to DB** | `lunch_eligible_at_time`, `lunch_reason_code`, `lunch_reason_notes`, `lunch_subscription` were added by 0026 but the DB columns don't exist. Current `models.py` omits them. |
| **`resources.py` references non-existent fields** | `AttendanceRecordResource` exports `meal_eligible_at_time`, `meal_reason_code`, `meal_reason_notes`, `meal_subscription_id` — none exist on the model or DB. Export would fail at runtime. |
| **`LunchSubscription` → `MealSubscription` was a delete+recreate, not a rename** | Migration 0028 `DeleteModel('LunchSubscription')` + `CreateModel('MealSubscription')`. Any data in `LunchSubscription` would have been lost (though the table didn't exist in DB due to the fake squash). |

**Impact:** the migration graph is internally consistent (`makemigrations --check` passes) but the **actual DB does not match the migration state**. Any migration that assumes `confirmed`/`lunch_*` columns exist will fail. This must be reconciled before adding Person FKs.

### 8.2 Wallet balance directly mutated in views

| Location | Operation |
|---|---|
| `views.py:1094` | `wallet.balance_iqd -= price; wallet.save()` (confirm, wallet mode, sufficient funds) |
| `views.py:1144` | `wallet.balance_iqd = projected_balance; wallet.save()` (confirm, allow_negative) |
| `views.py:1229` | `wallet.balance_iqd += refund_amount; wallet.save()` (reverse/refund) |
| `views_.py` | Same pattern (duplicate file). |

`WalletTransaction.objects.create(...)` is called directly from the same views (`views.py:1097`, `1146`, `1232`). This is **view-layer wallet math** with no service boundary — exactly what the finance-domain architecture forbids. It works today but is the highest-risk code to migrate.

### 8.3 `views.py` vs `views_.py` duplication

`views_.py` is a near-identical copy of `views.py` (1327 vs 1345 lines) missing only the `meal_bucket` filter block. It appears to be a stale backup/experiment. Any fix applied to `views.py` must also be applied to `views_.py` (or `views_.py` should be deleted).

### 8.4 `enable_postpaid` URL commented out

`urls.py:30` — the `enable_postpaid` route is commented out, but the view still exists. The dashboard may reference it; disabling it changes supervisor behavior.

## 9. What discount-related parts are incomplete?

| Part | State |
|---|---|
| `DiscountProfile` | Model exists (`models.py:729`), admin registered (`admin.py:688`) with `DiscountRuleInline`. But: only `name`/`is_active`/`notes` — no `type`, `value`, `priority`, `start_date`, `end_date`, or `academic_year` fields. |
| `DiscountRule` | Model exists (`models.py:741`) with `rule_type` (fixed/percent), `value_iqd`, `value_percent`, `priority`, `period_template`, `min_same_day_confirmed_meals`. But: no `DiscountAssignment` linking rules to Person/Family/Grade. |
| `MealProfile.discount_profile` FK | Exists (`models.py:816`) but **never used** in any charge path. `confirm_record` always sets `meal.discount_iqd = 0`. |
| Discount resolution | **Not implemented.** No `resolve_discount` function exists. `views.py:1057,1082` hardcode `discount_iqd = 0`. |
| `MealRecord.discount_iqd` | Field exists (`models.py:943`), always 0. |

**Conclusion:** the discount scaffolding (models + admin) exists but the **resolution engine is entirely missing**. Discounts are not applied anywhere in the live charge path.

## 10. Safest integration strategy into the new foundation

### Principle: wrap, don't rewrite

| Target | Strategy |
|---|---|
| **Person / StudentProfile** | Add a nullable `person`/`student_profile` FK alongside the legacy `student` FK (dual-FK). Backfill from `StudentProfile.legacy_student`. Flip read paths one at a time. Drop legacy FK last. |
| **StaffProfile** | `MealProfile` supervisor flags stay on the profile. Supervisor auth (`meal_supervisor` group) stays on `auth.User` until StaffProfile-based roles are built. Add `created_by_staff` (StaffProfile) alongside `created_by` (auth.User) on transactions; populate opportunistically. |
| **AcademicYear / Grade / Section / Enrollment** | Do **not** add these FKs to existing meal/wallet models yet. The legacy `Student.grade`/`Student.has_meal` mirrors remain the source of truth until `apps.academics` is rolled out and enrollments are backfilled. Lunch eligibility continues to use `MealSubscription` date ranges, not enrollment. |

### Order

1. **Reconcile migrations first** (fix the fake-squash drift) — see §13.
2. Add `person` FK to `Wallet` / `WalletTransaction` / `MealSubscription` (nullable, dual-FK).
3. Backfill `person` from `student.migrated_to.person`.
4. Introduce a `finance.charge()` / `finance.refund()` service boundary that the dashboard calls, replacing direct `wallet.balance_iqd` mutation.
5. Flip dashboard read paths from `student.h_code` to `person.display_code`.
6. Only after all reads use `person`, drop legacy `student` FKs.

## 11. What should be wrapped with adapters/services rather than rewritten?

| Code | Adapter |
|---|---|
| `_resolve_effective_meal_setup(student, day, period)` | Wrap as `meal.resolve_eligibility(student_profile, date)` adapter that internally calls the legacy function. New code calls the adapter; legacy dashboard keeps working. |
| `confirm_record` / `reverse_record` wallet logic | Extract into `finance.charge()` / `finance.refund()` services. The views call the services instead of mutating `wallet.balance_iqd` directly. Same behavior, service boundary. |
| `recalc_meal_flags_for_students` | Wrap as `meal.sync_has_meal_flag()` adapter. The signal keeps calling it until enrollment-based eligibility replaces it. |
| `ingest_match(h_code, ...)` | Add `ingest_match_by_person(person, ...)` adapter that resolves `Student` via `StudentProfile.legacy_student` for backward compat. |
| `Student.full_name()` / `Student.h_code` template accessors | Add a `legacy_student_adapter` property on `Person`/`StudentProfile` that delegates to the legacy `Student` during transition, so templates can switch to `person.display_code` incrementally. |
| `StudentResource` import/export | Keep until Person-based import is built; delegate internally. |

## 12. What should not be touched yet?

| Do not touch | Why |
|---|---|
| `RecognitionSettings`, `FaceEmbedding`, camera/scheduler infra | Person-agnostic; only FK target changes later. |
| `MealRecord` model structure | Working lifecycle; migrate FK only. |
| `MealProfile` / `MealProfilePeriod` | Working pricing; generalize later. |
| Discount models/resolution | Incomplete and not in the charge path; defer entirely until Finance + Lunch foundation is migrated. |
| `views_.py` | Stale duplicate; delete it only after confirming `views.py` is the live one, but do not edit it. |
| HTMX templates / dashboard UX | Preserve behavior; only swap data sources under the views. |
| `face_gallery/<h_code>/` filesystem layout | Keyed by `h_code`; remap to `person.code` only after Person migration is verified. |
| Legacy `Student` table | Keep through dual-FK phase; drop only after full verification. |

## 13. Recommended phased migration plan

### Phase 0: Reconcile migration drift (CRITICAL, before anything else)

1. Audit the actual DB schema vs migration state for `attendance_attendancerecord` (and all tables).
2. Create a corrective migration that brings the **migration state** in line with the **actual DB** — either:
   - Add the missing columns (`confirmed`, `lunch_*`) to the DB via a real migration, **or**
   - Officially remove them from the migration graph (if they were intentionally dropped) via an `AlterField`/`RemoveField` migration that matches the current `models.py`.
3. Fix `resources.py` to remove references to non-existent `meal_*` fields (or add them back to the model if needed).
4. Delete or reconcile `views_.py`.
5. Run `makemigrations --check` and verify the migration graph is honest.

### Phase 1: Person dual-FK (additive, no behavior change)

1. Add nullable `person` FK to `Wallet`, `WalletTransaction`, `MealSubscription`.
2. Data migration: backfill `person_id` from `student.migrated_to.person`.
3. Add `person` to admin search/list_display.
4. No read-path changes yet.

### Phase 2: Service boundary extraction

1. Create `apps.finance.services.charge()` / `refund()` / `check_balance()` that encapsulate the wallet-mutation + transaction-creation logic currently inline in `views.py`.
2. Refactor `confirm_record` / `reverse_record` to call these services.
3. Behavior must be **identical** — only the call site changes.

### Phase 3: Read-path migration

1. Switch dashboard row rendering from `student.h_code` / `student.full_name()` to `person.display_code` / `person.full_name`.
2. Switch search filters from `student__h_code__icontains` to `person__code__icontains` / `person__first_name__icontains`.
3. Switch serializers/resources from `student.h_code` to `person.display_code`.

### Phase 4: Eligibility adapter

1. Wrap `_resolve_effective_meal_setup` in a `meal.resolve_eligibility(student_profile, date)` service.
2. Keep the legacy function as the implementation; new code calls the service.

### Phase 5: Drop legacy FKs

1. Enforce `person` NOT NULL on `Wallet` / `WalletTransaction` / `MealSubscription`.
2. Drop legacy `student` FK.
3. Drop `Student` table (Phase 2 of the identity architecture).

## 14. Files/modules that need special attention

| File | Risk | Attention needed |
|---|---|---|
| `apps/attendance/views.py` | **HIGH** | Direct wallet mutation (lines 1094, 1144, 1229); 1345 lines of inline HTML + business logic. Extract services carefully. |
| `apps/attendance/views_.py` | **MEDIUM** | Stale duplicate; must be deleted or reconciled before any view change. |
| `apps/attendance/resources.py` | **HIGH** | References non-existent `meal_eligible_at_time`/`meal_reason_code`/`meal_subscription_id` fields. Export is broken. |
| `apps/attendance/migrations/0001_squashed_0021...py` | **CRITICAL** | Fake-applied squash; migration state ≠ DB state. |
| `apps/attendance/migrations/0026_...py` | **CRITICAL** | Added `lunch_*` fields never applied to DB. |
| `apps/attendance/migrations/0028_...py` | **MEDIUM** | `DeleteModel('LunchSubscription')` + `CreateModel('MealSubscription')` — destructive if re-run. |
| `apps/attendance/signals.py` | **LOW** | `post_save`/`post_delete` on `MealSubscription` — must keep working during FK migration. |
| `apps/attendance/utils/meal.py` | **LOW** | `recalc_meal_flags_for_students` — wrap as adapter, don't rewrite. |
| `apps/attendance/api.py` | **MEDIUM** | `h_code`-keyed ingest; needs `person`-keyed adapter. |
| `apps/attendance/admin.py` | **LOW** | Large file (2293 lines); `StudentAdmin` actions reference `h_code`; migrate incrementally. |
| `apps/attendance/serializers.py` | **MEDIUM** | `h_code` field sourced from `student.h_code`; switch to `person.display_code`. |
| `apps/attendance/utils/media_paths.py` | **MEDIUM** | `face_gallery/<h_code>/` filesystem layout; remap after Person migration. |

---

End of report.
